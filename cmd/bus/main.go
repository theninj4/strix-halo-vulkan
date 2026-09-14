// Command bus measures what a host read and a host write of a mapped Vulkan
// storage buffer actually cost on this part, and — the reason it exists —
// how that changes with how much has already been allocated.
//
//	go run ./cmd/bus -buffer 1200 -chunk 63 -pre 8
//
// The engine had carried one number since stage 3c: "reads out of the
// device-local host-visible arena run at 0.2 GB/s, writes at 11.5". It is
// right, and it is right only for a program that has allocated almost
// nothing. vk.NewBuffer asks for DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT
// and falls back to HOST_VISIBLE | HOST_COHERENT when that heap cannot serve
// the request, and on this device the first heap runs out at about **8 GB** —
// far below the 83.79 GiB it reports. The fallback is ordinary cached system
// memory, so a host read of it is **83x faster**:
//
//	pre=6 GB   read 361 ms   0.18 GB/s
//	pre=7 GB   read 4.7 ms  14.10 GB/s
//
// Writes do not move (29 GB/s either way), because they were already going
// through write-combining buffers.
//
// What that means in practice is that a microbenchmark and the real pipeline
// are on opposite sides of the line: the pipeline has 20.5 GB of weights
// resident before it allocates an activation arena, so every read-back it
// does is a fast one, while a benchmark that loads one stage measures the
// slow path and reports it as the cost of a read. See
// research/stage-9-head-and-tail.md.
package main

import (
	"flag"
	"fmt"
	"log"
	"time"

	"strix-halo-vulkan/vk"
)

const strixHaloDeviceID = 0x1586

func main() {
	buffer := flag.Int("buffer", 1200, "size of the buffer under test, in MB")
	chunk := flag.Int("chunk", 63, "bytes moved per measurement, in MB")
	pre := flag.Int("pre", 0, "GB of buffers to allocate before it, in 1 GB pieces")
	reps := flag.Int("reps", 3, "measurements")
	flag.Parse()

	inst, err := vk.NewInstance("bus")
	if err != nil {
		log.Fatal(err)
	}
	defer inst.Destroy()
	devs, err := inst.PhysicalDevices()
	if err != nil || len(devs) == 0 {
		log.Fatalf("no Vulkan devices: %v", err)
	}
	phys := &devs[0]
	for i := range devs {
		if devs[i].DeviceID == strixHaloDeviceID {
			phys = &devs[i]
		}
	}
	qf, err := phys.ComputeQueueFamily()
	if err != nil {
		log.Fatal(err)
	}
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{})
	if err != nil {
		log.Fatal(err)
	}
	defer dev.Destroy()
	fmt.Printf("device: %s\n", phys.Name)

	// The pressure buffers are never touched. What they change is which heap
	// has room when the buffer under test is allocated.
	for i := 0; i < *pre; i++ {
		b, err := dev.NewBuffer(1 << 30)
		if err != nil {
			log.Fatalf("pressure buffer %d of %d GB: %v", i, *pre, err)
		}
		defer b.Destroy()
	}
	buf, err := dev.NewBuffer(*buffer << 20)
	if err != nil {
		log.Fatal(err)
	}
	defer buf.Destroy()

	n := (*chunk << 20) / 4
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(i % 97)
	}
	mb := float64(n*4) / 1e6
	for rep := 0; rep < *reps; rep++ {
		t0 := time.Now()
		buf.WriteFloat32At(0, src)
		w := time.Since(t0)
		t0 = time.Now()
		out := buf.ReadFloat32At(0, n)
		r := time.Since(t0)
		fmt.Printf("pre %3d GB  buffer %5d MB  move %5.0f MB  write %9s (%5.1f GB/s)  read %9s (%6.2f GB/s)  sum %g\n",
			*pre, *buffer, mb, w.Round(time.Microsecond), mb/1e3/w.Seconds(),
			r.Round(time.Microsecond), mb/1e3/r.Seconds(), out[n-1])
	}
}
