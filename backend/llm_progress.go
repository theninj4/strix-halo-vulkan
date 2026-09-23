package backend

// The periodic progress line (LLMOptions.Progress).
//
// A completion leaves one line in the journal, at the end (logRun). At 128k
// tokens that end is two minutes of prefill and then however long the answer
// is, and with three slots interleaved there is no telling from the journal
// which conversation has the device and how fast it is going. So every
// interval the scheduler writes a line for each request in flight: where its
// prefill or generation has got to, and the rate over the last interval --
// not since it started, which would still be quoting the prefill a minute
// into the answer.

import (
	"fmt"
	"log"
	"time"
)

// jobProgress is a job's counters, as the scheduler loop leaves them. All of
// it is under llmSched.mu.
type jobProgress struct {
	// enter is when the job asked for a slot, granted when it got one, and
	// decoding when it submitted its first decode step (zero before).
	enter, granted, decoding time.Time
	// prefilled is how much of the prompt the slot holds, reused tokens
	// included, and steps how many decode steps have run.
	prefilled, steps int

	// What the last report saw (claim sets the first), so a rate is over
	// the interval.
	lastAt        time.Time
	lastPrefilled int
	lastSteps     int
	lastServed    time.Duration
}

// reportProgress is the reporter's goroutine; it stops with the loop.
func (s *llmSched) reportProgress(every time.Duration) {
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-s.stopped:
			return
		case now := <-t.C:
			for _, line := range s.progressLines(now, every/2) {
				log.Print(line)
			}
		}
	}
}

// progressLines is one line for each request that has been in flight for at
// least minAge. Anything younger is left to its completion line, which is
// seldom more than an interval away; without the floor, every voice command
// that happens to straddle a tick would be announced as well as logged.
func (s *llmSched) progressLines(now time.Time, minAge time.Duration) []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	var lines []string
	for _, j := range s.waiting {
		if now.Sub(j.progress.enter) < minAge {
			continue
		}
		lines = append(lines, fmt.Sprintf("%sllm [%s]: waiting for a slot, %v so far, prompt %d tokens",
			logID(j.ctx), j.class, now.Sub(j.progress.enter).Round(time.Second), len(j.prompt)))
	}
	live := 0
	for i := range s.slots {
		if s.slots[i].job != nil {
			live++
		}
	}
	for i := range s.slots {
		j := s.slots[i].job
		if j == nil || now.Sub(j.progress.enter) < minAge {
			continue
		}
		lines = append(lines, s.progressLine(j, now, live))
	}
	return lines
}

// progressLine describes one slot-holder and moves its baseline to now.
func (s *llmSched) progressLine(j *llmJob, now time.Time, live int) string {
	p := &j.progress
	where := fmt.Sprintf("[slot %d, %s]", j.slot, j.class)
	var what string
	if p.decoding.IsZero() {
		total := len(j.prompt)
		r := rate(p.prefilled-p.lastPrefilled, now.Sub(p.lastAt))
		what = fmt.Sprintf("prefill %d/%d tokens (%.0f%%), %.0f tok/s", p.prefilled, total,
			100*float64(p.prefilled)/float64(total), r)
		if r > 0 {
			eta := time.Duration(float64(total-p.prefilled) / r * float64(time.Second))
			what += fmt.Sprintf(", ~%v left", eta.Round(time.Second))
		}
	} else {
		from := p.lastAt
		if p.decoding.After(from) {
			from = p.decoding
		}
		// A step feeds back a token already sampled, so the tokens generated
		// are one more than the steps run.
		what = fmt.Sprintf("generating, %d tokens after a %d-token prompt, %.1f tok/s",
			p.steps+1, len(j.prompt), rate(p.steps-p.lastSteps, now.Sub(from)))
	}
	// With others in flight a rate is this request's share of the device,
	// not the device's speed, so the share is said alongside it.
	if live > 1 {
		if d := now.Sub(p.lastAt); d > 0 {
			what += fmt.Sprintf(", %.0f%% of the device", 100*float64(j.served-p.lastServed)/float64(d))
		}
	}
	p.lastAt, p.lastPrefilled, p.lastSteps, p.lastServed = now, p.prefilled, p.steps, j.served
	return fmt.Sprintf("%sllm %s: %s, %v in", logID(j.ctx), where, what, now.Sub(p.enter).Round(time.Second))
}
