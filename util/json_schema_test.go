package util

import (
	"encoding/json"
	"reflect"
	"testing"
)

type Person struct {
	Address Address `json:"address"`
	Name    string  `json:"name"          doc:"The person's full name"`
	Age     int     `json:"age,omitempty" doc:"Age in years"`
}

type Address struct {
	Street string `json:"street" doc:"Street address"`
	City   string `json:"city"   doc:"City name"`
}

func TestNewJSONSchema(t *testing.T) {
	schema, err := NewJSONSchema("name", "description", Address{})
	if err != nil {
		t.Errorf("NewJSONSchema() error = %v", err)
		return
	}
	s, err := json.Marshal(schema)
	if err != nil {
		t.Errorf("json.Marshal() error = %v", err)
		return
	}
	expected := `{"name":"name","description":"description","schema":{"additionalProperties":false,"type":"object","properties":{"city":{"type":"string","description":"City name"},"street":{"type":"string","description":"Street address"}},"required":["street","city"]},"strict":true}`
	// Parse both to compare structure
	var gotInterface, expectedInterface interface{}
	if err := json.Unmarshal(s, &gotInterface); err != nil {
		t.Errorf("Failed to unmarshal got: %v", err)
		return
	}
	if err := json.Unmarshal([]byte(expected), &expectedInterface); err != nil {
		t.Errorf("Failed to unmarshal expected: %v", err)
		return
	}
	if !reflect.DeepEqual(gotInterface, expectedInterface) {
		t.Errorf("Mismatch!\n got = %s\n want= %s", s, expected)
	}
}

func TestNewJSONSchemaPartial(t *testing.T) {
	tests := []struct {
		name   string
		input  any
		output string
	}{
		{
			name:   "Empty struct",
			input:  struct{}{},
			output: `{"additionalProperties":false,"type":"object","properties":{},"required":[]}`,
		},
		{
			name:   "Simple struct",
			input:  Address{},
			output: `{"additionalProperties":false,"type":"object","properties":{"city":{"type":"string","description":"City name"},"street":{"type":"string","description":"Street address"}},"required":["street","city"]}`,
		},
		{
			name:   "Complex struct",
			input:  Person{},
			output: `{"additionalProperties":false,"type":"object","properties":{"address":{"additionalProperties":false,"type":"object","properties":{"city":{"type":"string","description":"City name"},"street":{"type":"string","description":"Street address"}},"required":["street","city"]},"age":{"type":"integer","description":"Age in years"},"name":{"type":"string","description":"The person's full name"}},"required":["address","name"]}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			schema, err := NewJSONSchemaPartial(tt.input)
			if err != nil {
				t.Errorf("NewJSONSchemaPartial() error = %v", err)
				return
			}
			s, err := json.Marshal(schema)
			if err != nil {
				t.Errorf("json.Marshal() error = %v", err)
				return
			}
			// Parse both to compare structure
			var gotInterface, expectedInterface interface{}
			if err := json.Unmarshal(s, &gotInterface); err != nil {
				t.Errorf("Failed to unmarshal got: %v", err)
				return
			}
			if err := json.Unmarshal([]byte(tt.output), &expectedInterface); err != nil {
				t.Errorf("Failed to unmarshal expected: %v", err)
				return
			}
			if !reflect.DeepEqual(gotInterface, expectedInterface) {
				t.Errorf("Mismatch!\n got = %+v\n want= %+v", gotInterface, expectedInterface)
			}
		})
	}
}
