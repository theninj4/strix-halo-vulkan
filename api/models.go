package api

type ModelsResponse struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

// Model represents a single model in the list of models.
type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`   // "model"
	OwnedBy string `json:"owned_by"` // "organization_owner"
}
