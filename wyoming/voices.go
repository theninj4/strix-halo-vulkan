package wyoming

import (
	"sort"
	"strings"
)

// This file is the one place in the package that knows what a voice name
// means, because Wyoming's `info` message asks a question OpenAI's
// /v1/models does not: what language does this voice speak?
//
// Kokoro answers it in the filename. A pack is `<language><gender>_<name>` --
// `af_heart` is American English, female, "heart" -- and the letters are
// hexgrad's own, the same ones misaki's pipeline switches its front end on.
// Nothing in the checkpoint states them, so the table below is a fact about
// upstream's naming and is checked against the voice list the model actually
// loaded rather than assumed: a pack whose prefix is not here is reported
// with no language, and how that is handled is Voices' business.

// language is one of kokoro's nine, as the tag a client sees and the name a
// person reads.
type language struct {
	tag  string
	name string
}

// languages maps a voice name's first letter onto the language its pack
// speaks. The tags are hyphenated BCP 47, which is the spelling Home
// Assistant's pipeline language menu uses.
var languages = map[byte]language{
	'a': {"en-US", "American English"},
	'b': {"en-GB", "British English"},
	'e': {"es", "Spanish"},
	'f': {"fr-fr", "French"},
	'h': {"hi", "Hindi"},
	'i': {"it", "Italian"},
	'j': {"ja", "Japanese"},
	'p': {"pt-br", "Brazilian Portuguese"},
	'z': {"zh", "Mandarin Chinese"},
}

// genders maps the second letter. It is only ever used to build a label.
var genders = map[byte]string{'f': "female", 'm': "male"}

// englishTags are the languages this server's grapheme-to-phoneme front end
// can actually reach from text, which is not the same set as the voices it
// can speak with.
//
// That distinction is the reason Voices takes an `all` argument. Kokoro's 54
// packs cover nine languages; misaki's English G2P is the only one ported
// (SPEECH.md T5), so a request naming `jf_alpha` gets a Japanese voice
// reading English phonemes -- which is a recognisable accent rather than
// Japanese. Over HTTP that is the caller's problem, because
// /v1/audio/speech also takes IPA directly and a caller with its own lexicon
// is entitled to every pack. Over Wyoming there is no phoneme field: Home
// Assistant sends text, so advertising a voice this server cannot pronounce
// is advertising a bug.
var englishTags = map[string]bool{"en-US": true, "en-GB": true}

// Voices describes voice packs for an `info` message.
//
// The names are the backend's own, in its order; what comes back is sorted by
// name so the client's menu is stable across restarts. With all false -- the
// default -- only the packs whose language the front end can pronounce are
// described; with it true every pack is, each tagged with the language it
// actually speaks, which is what a deployment that sends phonemes or that has
// another lexicon wants.
//
// A name whose prefix is not one of kokoro's is described with no language.
// Home Assistant groups its voice menu by language and would never show it,
// which is the right outcome for a pack this server cannot place.
func Voices(names []string, all bool) []TtsVoice {
	sorted := append([]string(nil), names...)
	sort.Strings(sorted)

	out := make([]TtsVoice, 0, len(sorted))
	for _, name := range sorted {
		lang, ok := voiceLanguage(name)
		if !all && (!ok || !englishTags[lang.tag]) {
			continue
		}
		var tags []string
		if ok {
			tags = []string{lang.tag}
		}
		desc := describeVoice(name)
		out = append(out, TtsVoice{
			Artifact: Artifact{
				Name:        name,
				Description: &desc,
				Attribution: kokoroAttribution,
				Installed:   true,
			},
			Languages: tags,
			// Explicitly absent: a pack is one speaker. A blend of packs
			// is spelled in the voice name -- "af_bella,af_sky" -- and
			// reaches backend.TTS through the same field, so a client that
			// wants one writes it rather than selecting it here.
			Speakers: nil,
		})
	}
	return out
}

// Languages are the language tags a set of described voices covers, sorted.
// Home Assistant builds its pipeline's language menu out of exactly this.
func Languages(voices []TtsVoice) []string {
	seen := map[string]bool{}
	var out []string
	for _, v := range voices {
		for _, tag := range v.Languages {
			if !seen[tag] {
				seen[tag] = true
				out = append(out, tag)
			}
		}
	}
	sort.Strings(out)
	return out
}

// voiceLanguage reads a pack name's prefix.
func voiceLanguage(name string) (language, bool) {
	if len(name) < 3 || name[2] != '_' {
		return language{}, false
	}
	lang, ok := languages[name[0]]
	return lang, ok
}

// describeVoice is the label a person picks from: the pack's given name, then
// what it is. `af_heart` becomes "Heart (American English, female)", which
// sorts and reads better in a menu than the filename and still lets someone
// who knows the filename find it.
func describeVoice(name string) string {
	lang, ok := voiceLanguage(name)
	if !ok {
		return name
	}
	given := name[3:]
	if given == "" {
		return name
	}
	label := strings.ToUpper(given[:1]) + given[1:]
	if gender, ok := genders[name[1]]; ok {
		return label + " (" + lang.name + ", " + gender + ")"
	}
	return label + " (" + lang.name + ")"
}

// kokoroAttribution and parakeetAttribution are who made the two checkpoints.
// Every artifact in an `info` message needs one, and the reference client
// will not decode an artifact without it.
var (
	kokoroAttribution = Attribution{
		Name: "hexgrad",
		URL:  "https://huggingface.co/hexgrad/Kokoro-82M",
	}
	parakeetAttribution = Attribution{
		Name: "NVIDIA",
		URL:  "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",
	}
)

// asrLanguages are the languages parakeet-tdt-0.6b-v3 is trained on: the 25
// European languages of its v3 release.
//
// They are listed rather than read from the checkpoint because the checkpoint
// does not say. Its tokenizer is a single multilingual vocabulary and its
// decode reports no language, which is also why backend.STT ignores the
// request's language hint -- so this list describes what the model was
// trained for and is not a set it validates against. Home Assistant uses it
// for one thing: deciding which pipeline languages may select this engine.
var asrLanguages = []string{
	"bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu",
	"it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk",
}
