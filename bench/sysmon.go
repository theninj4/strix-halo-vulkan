package bench

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// ClockStats summarises the GPU's shader clock and package power over one
// timed measurement. It exists because this part idles at 636 MHz out of a
// 2900 MHz maximum (see /sys/class/drm/cardN/device/pp_dpm_sclk), so a
// measurement taken before the clocks ramp reports the kernel as several
// times slower than it is — indistinguishable, in a results table, from a
// genuinely slow kernel. Recording the clock alongside every number makes
// that failure mode visible instead of silent.
type ClockStats struct {
	SclkMHz    float64 // mean shader clock over the measurement
	SclkMinMHz float64
	SclkMaxMHz float64
	PowerW     float64 // mean package power (amdgpu's PPT sensor)
	Samples    int
}

// Instruments conditions and observes the GPU around each timed
// measurement: Warm pulls the clocks up off idle before timing starts, and
// Watch samples sclk/power while the timed batch runs. Both halves degrade
// independently — a machine without amdgpu's hwmon counters still gets
// clock warming (with zeroed stats), and a zero warm duration disables
// warming while keeping the sampling.
type Instruments struct {
	warmer    *clockWarmer
	freqPath  string // hwmon freqN_input, Hz
	powerPath string // hwmon powerN_average, microwatts
	interval  time.Duration
	// warmFor bounds how long Warm may run. It is a deadline rather than a
	// fixed duration: warming stops as soon as the clock reaches
	// targetMHz, so a GPU that is already hot costs one short burst.
	warmFor time.Duration
	// targetMHz is the top clock this GPU advertises (from pp_dpm_sclk), or
	// 0 if it couldn't be read — in which case Warm has no way to tell it
	// has succeeded and simply runs out its deadline.
	targetMHz float64
}

// instruments is the process-wide instrumentation TimeDispatch consults.
// A package-level hook rather than a parameter because every op family
// calls TimeDispatch and none of them has any business knowing about
// clock conditioning; nil (the default) leaves timing behaviour unchanged.
var instruments *Instruments

// SetInstruments installs the instrumentation TimeDispatch uses. Passing
// nil restores the uninstrumented behaviour.
func SetInstruments(in *Instruments) { instruments = in }

// NewInstruments builds the instrumentation. warmFor is how long to run
// ALU-heavy work before each timed measurement (0 disables it); sampleEvery
// is the clock-sampling period. Discovery of the sysfs counters is
// best-effort: if they aren't found, the returned Instruments still warms
// and simply reports no clock data.
func NewInstruments(dev *vk.Device, warmFor, sampleEvery time.Duration) (*Instruments, error) {
	in := &Instruments{interval: sampleEvery}
	if in.interval <= 0 {
		in.interval = time.Millisecond
	}
	in.freqPath, in.powerPath = findAMDGPUSensors()
	in.warmFor = warmFor
	in.targetMHz = maxDPMSclkMHz()
	if warmFor > 0 {
		w, err := newClockWarmer(dev)
		if err != nil {
			return nil, err
		}
		in.warmer = w
	}
	return in, nil
}

// Describe reports what instrumentation is actually active, so a run's
// output records whether its numbers carry clock data or not.
func (in *Instruments) Describe() string {
	if in == nil {
		return "none"
	}
	var parts []string
	switch {
	case in.warmer == nil:
		parts = append(parts, "no clock warmup")
	case in.targetMHz > 0:
		parts = append(parts, fmt.Sprintf("clock warmup to %.0f MHz (%.0f%% of %.0f MHz max), up to %s",
			in.targetMHz*clockWarmerTargetFrac, clockWarmerTargetFrac*100, in.targetMHz, in.warmFor))
	default:
		parts = append(parts, fmt.Sprintf("clock warmup for %s (no pp_dpm_sclk target found)", in.warmFor))
	}
	if in.freqPath != "" {
		parts = append(parts, fmt.Sprintf("sclk/power sampling every %s from %s", in.interval, filepath.Dir(in.freqPath)))
	} else {
		parts = append(parts, "no amdgpu hwmon counters found")
	}
	return strings.Join(parts, ", ")
}

// Destroy releases the warmup pipeline.
func (in *Instruments) Destroy() {
	if in != nil && in.warmer != nil {
		in.warmer.destroy()
	}
}

// Warm drives the GPU with ALU-heavy work until its shader clock reaches
// clockWarmerTargetFrac of the advertised maximum, or until the configured
// deadline expires.
//
// Closed-loop rather than a fixed duration because the ramp time is not
// constant: it depends on how long the GPU has been idle, which in this
// harness depends on how much host-side work (generating and quantizing
// hundreds of MB of weights) preceded the measurement. A fixed warmup long
// enough for the worst case would be wasted on every other case, and one
// tuned for the common case leaves the expensive cases measured mid-ramp —
// which is exactly the artefact that made the original sweep's numbers
// non-monotonic in size.
func (in *Instruments) Warm() error {
	if in == nil || in.warmer == nil {
		return nil
	}
	deadline := time.Now().Add(in.warmFor)
	for {
		if err := in.warmer.burst(clockWarmerBurst); err != nil {
			return err
		}
		if !time.Now().Before(deadline) {
			return nil
		}
		if in.targetMHz <= 0 || in.freqPath == "" {
			continue
		}
		if sclk, _, ok := in.sample(); ok && sclk >= in.targetMHz*clockWarmerTargetFrac {
			return nil
		}
	}
}

// Watch starts sampling and returns a function that stops sampling and
// returns the aggregate. The returned function must be called exactly once.
func (in *Instruments) Watch() func() ClockStats {
	if in == nil || in.freqPath == "" {
		return func() ClockStats { return ClockStats{} }
	}
	done := make(chan struct{})
	out := make(chan ClockStats, 1)
	go func() {
		st := ClockStats{SclkMinMHz: math.Inf(1)}
		var sclkSum, powerSum float64
		ticker := time.NewTicker(in.interval)
		defer ticker.Stop()
		for {
			if sclk, power, ok := in.sample(); ok {
				st.Samples++
				sclkSum += sclk
				powerSum += power
				st.SclkMinMHz = math.Min(st.SclkMinMHz, sclk)
				st.SclkMaxMHz = math.Max(st.SclkMaxMHz, sclk)
			}
			select {
			case <-done:
				if st.Samples > 0 {
					st.SclkMHz = sclkSum / float64(st.Samples)
					st.PowerW = powerSum / float64(st.Samples)
				} else {
					st.SclkMinMHz = 0
				}
				out <- st
				return
			case <-ticker.C:
			}
		}
	}()
	return func() ClockStats {
		close(done)
		return <-out
	}
}

func (in *Instruments) sample() (sclkMHz, powerW float64, ok bool) {
	hz, err := readSysfsUint(in.freqPath)
	if err != nil {
		return 0, 0, false
	}
	sclkMHz = float64(hz) / 1e6
	if in.powerPath != "" {
		if uw, err := readSysfsUint(in.powerPath); err == nil {
			powerW = float64(uw) / 1e6
		}
	}
	return sclkMHz, powerW, true
}

// findAMDGPUSensors locates amdgpu's shader-clock and power sensors. The
// hwmon index and even the freqN slot vary by kernel version and by how
// many DRM cards are present, so match on the *_label files rather than
// hard-coding paths. Assumes a single amdgpu GPU, which is the case on this
// APU-only machine; with a discrete card also present this would need to
// match the card against the Vulkan device's PCI address.
func findAMDGPUSensors() (freqPath, powerPath string) {
	labels, _ := filepath.Glob("/sys/class/drm/card*/device/hwmon/hwmon*/freq*_label")
	for _, labelPath := range labels {
		label, err := os.ReadFile(labelPath)
		if err != nil || strings.TrimSpace(string(label)) != "sclk" {
			continue
		}
		candidate := strings.TrimSuffix(labelPath, "_label") + "_input"
		if _, err := os.Stat(candidate); err != nil {
			continue
		}
		freqPath = candidate
		dir := filepath.Dir(labelPath)
		for _, name := range []string{"power1_average", "power1_input"} {
			if _, err := os.Stat(filepath.Join(dir, name)); err == nil {
				powerPath = filepath.Join(dir, name)
				break
			}
		}
		return freqPath, powerPath
	}
	return "", ""
}

// maxDPMSclkMHz reads the highest shader-clock DPM level the driver
// advertises, e.g. the "2: 2900Mhz" line of
// /sys/class/drm/cardN/device/pp_dpm_sclk. Returns 0 if unavailable.
func maxDPMSclkMHz() float64 {
	paths, _ := filepath.Glob("/sys/class/drm/card*/device/pp_dpm_sclk")
	var max float64
	for _, path := range paths {
		b, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		for _, line := range strings.Split(string(b), "\n") {
			// Lines look like "2: 2900Mhz *"; take the numeric prefix of
			// the field ending in "Mhz".
			for _, field := range strings.Fields(line) {
				lower := strings.ToLower(field)
				if !strings.HasSuffix(lower, "mhz") {
					continue
				}
				v, err := strconv.ParseFloat(strings.TrimSuffix(lower, "mhz"), 64)
				if err == nil && v > max {
					max = v
				}
			}
		}
	}
	return max
}

func readSysfsUint(path string) (uint64, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	return strconv.ParseUint(strings.TrimSpace(string(b)), 10, 64)
}

// clockWarmerReps is how many multiply-adds per accumulator chain each
// warmup dispatch issues. Sized so one dispatch is well under a
// millisecond — short enough to stop promptly at the requested duration,
// long enough that per-dispatch overhead isn't most of the work.
const clockWarmerReps = 20000

// clockWarmerGroups is the warmup dispatch's wave count: enough waves to
// occupy all 40 CUs several times over, so the GPU sees a real load rather
// than a trickle.
const clockWarmerGroups = 640

// clockWarmerBurst is how long to drive the GPU between clock checks. Long
// enough that the sysfs read is a negligible fraction of the work, short
// enough to stop promptly once the target clock is reached.
const clockWarmerBurst = 20 * time.Millisecond

// clockWarmerTargetFrac is the fraction of the advertised maximum clock
// that counts as warmed. Not 1.0: the reported clock fluctuates by a few MHz
// at the top DPM level, so demanding the exact maximum would never
// terminate early.
const clockWarmerTargetFrac = 0.97

// clockWarmer runs the memory-free fp32 FMA microbenchmark in a loop to
// pull the GPU's shader clock up from idle before a measurement. It uses
// the ALU peak kernel rather than a memory-bound one deliberately: sclk is
// what these benchmarks are sensitive to, and a bandwidth-bound workload
// biases the driver toward raising memory clocks instead.
type clockWarmer struct {
	mod  *vk.ShaderModule
	in   *vk.Buffer
	out  *vk.Buffer
	pipe *vk.ComputePipeline
	pc   []byte
}

func newClockWarmer(dev *vk.Device) (*clockWarmer, error) {
	w := &clockWarmer{}
	var err error
	if w.mod, err = dev.NewShaderModule(shaders.ALUPeakFMA); err != nil {
		return nil, err
	}
	if w.in, err = dev.NewBuffer(aluPeakInputFloats * 4); err != nil {
		w.destroy()
		return nil, err
	}
	w.in.WriteFloat32(aluPeakOperands())
	if w.out, err = dev.NewBuffer(clockWarmerGroups * aluPeakLocalSize * 4); err != nil {
		w.destroy()
		return nil, err
	}
	w.pipe, err = dev.NewPipeline(w.mod, vk.PipelineSpec{
		Buffers:          []*vk.Buffer{w.in, w.out},
		PushConstantSize: 4,
	})
	if err != nil {
		w.destroy()
		return nil, err
	}
	w.pc = newPC().U32(clockWarmerReps).Bytes()
	return w, nil
}

// burst runs warmup dispatches for approximately d.
func (w *clockWarmer) burst(d time.Duration) error {
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		if _, err := w.pipe.DispatchTimed(clockWarmerGroups, 1, 1, 1, w.pc); err != nil {
			return err
		}
	}
	return nil
}

func (w *clockWarmer) destroy() {
	if w.pipe != nil {
		w.pipe.Destroy()
	}
	if w.out != nil {
		w.out.Destroy()
	}
	if w.in != nil {
		w.in.Destroy()
	}
	if w.mod != nil {
		w.mod.Destroy()
	}
}
