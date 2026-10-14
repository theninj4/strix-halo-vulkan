// Command breeze synthesises speech with Breeze-TTS-2 on the device: text,
// optionally an instruction describing the voice, a WAV out, and where the
// time went (research/tts-instruct-vertical.md).
//
//	go run ./cmd/breeze -text 'Good morning everyone!' -instruction 'An enthusiastic radio DJ.' -o out.wav
//	go run ./cmd/breeze -text 'Hello.' -cfg 1 -seed 3 -o out.wav     # no instruction: a random speaker
//	go run ./cmd/breeze -text '...' -ref voice.wav -ref-text '...' -o out.wav   # clone (B8)
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"runtime/pprof"
	"time"

	"strix-halo-vulkan/audio"
	"strix-halo-vulkan/breeze"
	"strix-halo-vulkan/vk"
)

func main() {
	log.SetFlags(0)
	dir := flag.String("model", "models/Breeze-TTS-2", "checkpoint directory")
	text := flag.String("text", "The quick brown fox jumps over the lazy dog.", "text to speak")
	instruction := flag.String("instruction", "", "a description of the voice and its delivery")
	cfg := flag.Float64("cfg", 0, "guidance scale; 0 takes 4 with an instruction and 1 without")
	seed := flag.Uint64("seed", 1, "sampling seed")
	temp := flag.Float64("temp", 0, "sampling temperature; 0 takes 0.9")
	chunk := flag.Int("chunk", 2, "frames handed to the codec at once")
	out := flag.String("o", "out/breeze.wav", "write the 24 kHz WAV here")
	reps := flag.Int("reps", 1, "runs; the stats of each are printed")
	cpuprof := flag.String("cpuprofile", "", "write a CPU profile of the runs here")
	ref := flag.String("ref", "", "a reference clip to clone (16-bit PCM WAV, any rate); needs -ref-text")
	refText := flag.String("ref-text", "", "the reference clip's exact transcript")
	hostSampler := flag.Bool("host-sampler", false, "sample on the host after every pass (the B4 loop, the control)")
	fp16 := flag.Bool("fp16", false, "stage the layers in fp16, not int8 (the control)")
	flag.Parse()

	dev, done, err := openDevice()
	if err != nil {
		log.Fatal(err)
	}
	defer done()
	t0 := time.Now()
	mo := breeze.DefaultModelOptions()
	if *fp16 {
		mo.Decoder.Bank = breeze.BankFP16
	}
	m, err := breeze.LoadModel(dev, *dir, mo)
	if err != nil {
		log.Fatal(err)
	}
	defer m.Destroy()
	log.Printf("staged in %v: %.2f GB on the device", time.Since(t0).Round(time.Millisecond), float64(m.Bytes())/1e9)

	req := &breeze.Request{Text: *text, Instruction: *instruction}
	if *ref != "" {
		clip, err := audio.ReadWAV(*ref)
		if err != nil {
			log.Fatal(err)
		}
		if clip, err = audio.Resample(clip, breeze.SampleRate); err != nil {
			log.Fatal(err)
		}
		enc, err := breeze.LoadCodecEncoder(*dir)
		if err != nil {
			log.Fatal(err)
		}
		t0 := time.Now()
		if req.RefFrames, err = enc.Encode(clip.Samples, nil); err != nil {
			log.Fatal(err)
		}
		req.RefText = *refText
		log.Printf("reference: %.2f s -> %d frames in %v", clip.Duration(), len(req.RefFrames), time.Since(t0).Round(time.Millisecond))
	}
	if *cpuprof != "" {
		f, err := os.Create(*cpuprof)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	for r := 0; r < *reps; r++ {
		var wav []float32
		st, err := m.Generate(context.Background(), req, breeze.GenOptions{CFG: *cfg, Temperature: *temp, Seed: *seed, Chunk: *chunk, HostSampler: *hostSampler},
			func(s []float32) error { wav = append(wav, s...); return nil })
		if err != nil {
			log.Fatal(err)
		}
		sec := float64(len(wav)) / breeze.SampleRate
		log.Printf("%d frames = %.2f s in %v (%.2fx real time), first audio %v", st.Frames, sec,
			st.Total.Round(time.Millisecond), sec/st.Total.Seconds(), st.FirstAudio.Round(time.Millisecond))
		log.Printf("  prompt %d: encode %v prefill %v | backbone %v depth %v sample %v codec %v",
			st.Prompt, ms(st.Encode), ms(st.Prefill), ms(st.Backbone), ms(st.Depth), ms(st.Sample), ms(st.Codec))
		if st.Frames > 0 {
			f := float64(st.Frames)
			log.Printf("  a frame: backbone %.2f ms, depth %.2f ms, sample %.2f ms, device frame %.2f ms, codec %.2f ms, wall %.2f ms",
				st.Backbone.Seconds()*1e3/f, st.Depth.Seconds()*1e3/f, st.Sample.Seconds()*1e3/f, st.Frame.Seconds()*1e3/f,
				st.Codec.Seconds()*1e3/f, (st.Total-st.Encode-st.Prefill).Seconds()*1e3/f)
		}
		if r == 0 && *out != "" {
			if err := (&audio.Clip{Samples: wav, Rate: breeze.SampleRate}).WriteWAV(*out); err != nil {
				log.Fatal(err)
			}
		}
	}
}

func ms(d time.Duration) string { return fmt.Sprintf("%.1fms", float64(d.Microseconds())/1000) }

// openDevice picks this machine's iGPU, or the first device with a compute
// queue.
func openDevice() (*vk.Device, func(), error) {
	inst, err := vk.NewInstance("breeze")
	if err != nil {
		return nil, nil, err
	}
	devices, err := inst.PhysicalDevices()
	if err != nil || len(devices) == 0 {
		inst.Destroy()
		return nil, nil, fmt.Errorf("no Vulkan devices: %v", err)
	}
	phys := &devices[0]
	for i := range devices {
		if devices[i].DeviceID == 0x1586 {
			phys = &devices[i]
			break
		}
	}
	qf, err := phys.ComputeQueueFamily()
	if err != nil {
		inst.Destroy()
		return nil, nil, err
	}
	sgs, err := phys.SubgroupSizeControl()
	if err != nil {
		inst.Destroy()
		return nil, nil, err
	}
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported})
	if err != nil {
		inst.Destroy()
		return nil, nil, err
	}
	return dev, func() { dev.Destroy(); inst.Destroy() }, nil
}
