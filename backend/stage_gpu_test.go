package backend

import (
	"os"
	"testing"

	"strix-halo-vulkan/ace/lm"
	apipe "strix-halo-vulkan/ace/pipeline"
	"strix-halo-vulkan/ace/plan"
	qpipe "strix-halo-vulkan/qimage/pipeline"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/qwen"
)

// TestStagingSkipsTheQueue is the claim the swap slot's loads rest on:
// staging image and music is allocation, pipeline creation and mapped
// writes, with no submit, so it runs without Device.Do and speech keeps
// the device through a 15-28 s load (as video's stagings do, VIDEO.md M9).
// Each stages the served configuration (int8, three edit references, the
// LM) and is freed before the next.
func TestStagingSkipsTheQueue(t *testing.T) {
	if os.Getenv("STAGE_GPU") == "" {
		t.Skip("stages ~21 GB twice; set STAGE_GPU=1")
	}
	d, err := OpenDevice("stage-test")
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	defer d.Close()
	// The counter is live: a queue call moves it.
	before := vk.Submits()
	if err := d.dev.WaitIdle(); err != nil || vk.Submits() == before {
		t.Fatalf("vk.Submits did not count a WaitIdle (%v)", err)
	}

	t.Run("image", func(t *testing.T) {
		before := vk.Submits()
		p, err := qpipe.New(d.dev, qpipe.Options{Model: "../models/Qwen-Image-2.1", Refs: 3, Bank: qwen.BankQ8})
		if err != nil {
			t.Fatal(err)
		}
		n := vk.Submits() - before
		p.Destroy()
		if n != 0 {
			t.Fatalf("staging submitted %d times", n)
		}
	})
	t.Run("music", func(t *testing.T) {
		before := vk.Submits()
		p, err := apipe.New(d.dev, apipe.DefaultDirs("../models"), plan.MaxSeconds)
		if err != nil {
			t.Fatal(err)
		}
		defer p.Destroy()
		if err := p.LoadLM(d.dev, apipe.LMDir("../models"), lm.DefaultOptions()); err != nil {
			t.Fatal(err)
		}
		if n := vk.Submits() - before; n != 0 {
			t.Fatalf("staging submitted %d times", n)
		}
	})
}
