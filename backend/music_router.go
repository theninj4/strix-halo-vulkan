package backend

import (
	"context"

	"strix-halo-vulkan/api"
)

// MusicRouter serves /v1/music from ACE-Step and MiniMax-Music3 together
// (research/music-minimax.md N-d4): a request whose model names MiniMax
// (IsMusicMM3) runs MiniMax-Music3, and everything else -- no model, a
// client's own default, a name nothing matches -- runs ACE's resolveModel,
// as before (N-d1: ACE stays the default). Either may be absent: with one
// backend every request goes to it. One job queue serves both, and each is
// its own entry in the swap slot (N-d2).
type MusicRouter struct {
	ACE *Music
	MM3 *MusicMM3
}

func (r *MusicRouter) pick(model string) api.MusicBackend {
	switch {
	case r.ACE == nil:
		return r.MM3
	case r.MM3 != nil && IsMusicMM3(model):
		return r.MM3
	}
	return r.ACE
}

// Models is ACE's (the default first) then MiniMax's.
func (r *MusicRouter) Models() []api.Model {
	var out []api.Model
	if r.ACE != nil {
		out = append(out, r.ACE.Models()...)
	}
	if r.MM3 != nil {
		out = append(out, r.MM3.Models()...)
	}
	return out
}

// MusicInfo is the default backend's.
func (r *MusicRouter) MusicInfo() api.MusicInfo { return r.pick("").MusicInfo() }

// PlanMusic plans on the backend the model picks.
func (r *MusicRouter) PlanMusic(req *api.MusicRequest) (*api.MusicPlan, error) {
	return r.pick(req.Model).PlanMusic(req)
}

// GenerateMusic runs on the backend the model picks.
func (r *MusicRouter) GenerateMusic(ctx context.Context, req *api.MusicRequest, mp *api.MusicPlan, dst string, progress func(api.MusicProgress)) error {
	return r.pick(req.Model).GenerateMusic(ctx, req, mp, dst, progress)
}

// Close frees both.
func (r *MusicRouter) Close() {
	if r.ACE != nil {
		r.ACE.Close()
	}
	if r.MM3 != nil {
		r.MM3.Close()
	}
}
