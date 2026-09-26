package pipeline

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"

	"strix-halo-vulkan/h3/plan"
)

// FFmpeg is the muxer (decision 7): we write no encoder of our own.
var FFmpeg = "ffmpeg"

// WriteMP4 muxes a result to an H.264/AAC mp4 at path: the frames piped as
// raw rgb24 on stdin, the soundtrack from a float32 file beside the output,
// removed after. The stream is cut to the video's length.
func WriteMP4(path string, res *Result) error {
	if len(res.Video) == 0 {
		return fmt.Errorf("h3: no frames to mux")
	}
	pcm, err := os.CreateTemp(filepath.Dir(path), ".h3-audio-*.f32")
	if err != nil {
		return err
	}
	defer os.Remove(pcm.Name())
	w := bufio.NewWriterSize(pcm, 1<<20)
	var b [8]byte
	for i := range res.Left {
		binary.LittleEndian.PutUint32(b[:4], math.Float32bits(res.Left[i]))
		binary.LittleEndian.PutUint32(b[4:], math.Float32bits(res.Right[i]))
		w.Write(b[:])
	}
	if err := w.Flush(); err != nil {
		pcm.Close()
		return err
	}
	pcm.Close()

	cmd := exec.Command(FFmpeg, "-loglevel", "error", "-y",
		"-f", "rawvideo", "-pix_fmt", "rgb24", "-s", fmt.Sprintf("%dx%d", res.Width, res.Height),
		"-r", strconv.Itoa(plan.FPS), "-i", "pipe:0",
		"-f", "f32le", "-ar", strconv.Itoa(res.SampleRate), "-ac", "2", "-i", pcm.Name(),
		"-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
		"-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", path)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	stderr, err := os.CreateTemp("", "h3-ffmpeg-*.log")
	if err != nil {
		return err
	}
	defer os.Remove(stderr.Name())
	defer stderr.Close()
	cmd.Stderr = stderr
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("h3: %s: %w", FFmpeg, err)
	}
	var werr error
	for _, f := range res.Video {
		if _, werr = stdin.Write(f); werr != nil {
			break
		}
	}
	stdin.Close()
	if err := cmd.Wait(); err != nil {
		msg, _ := os.ReadFile(stderr.Name())
		return fmt.Errorf("h3: %s: %w: %s", FFmpeg, err, msg)
	}
	return werr
}
