package gemma4

// The decisions chat as Gemma 4's chat template renders it with the
// generation prompt and thinking off (R2): a system turn, a user turn, and
// the model turn opened on an empty thought, so the next token is the
// answer. Transcribed from chat_template.jinja, which is the same file in
// google/gemma-4-26B-A4B-it and in Rune (18,683 bytes); TestPromptTokens
// holds it to HF's apply_chat_template.
//
// The template trims both turns' content (`| trim`). A decisions user turn
// starts with "SHARED STATE" and ends with "only.", and its system prompt is
// fixed, so the trim never applies and is not reproduced.
const (
	turnSystem = "<bos><|turn>system\n"
	turnUser   = "<turn|>\n<|turn>user\n"
	turnModel  = "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
)

// DecisionPrompt is the rendered text of one question's chat.
func DecisionPrompt(system, user string) string {
	return turnSystem + system + turnUser + user + turnModel
}
