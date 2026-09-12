package util

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
)

// JSONSchema represents a complete JSON schema including name and description.
type JSONSchema struct {
	Schema      *JSONSchemaType `json:"schema,omitempty"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Strict      bool            `json:"strict,omitempty"`
}

// JSONSchemaType represents a partial JSON schema type.
// It is one-of JSONSchemaObject, JSONSchemaArray, or JSONSchemaPrimitive.
type JSONSchemaType any

// JSONSchemaObject represents a JSON schema object type.
type JSONSchemaObject struct {
	Properties           map[string]JSONSchemaType `json:"properties"`
	Type                 string                    `json:"type"`
	Required             []string                  `json:"required"`
	AdditionalProperties bool                      `json:"additionalProperties"`
}

// JSONSchemaArray represents a JSON schema array type.
type JSONSchemaArray struct {
	Items *JSONSchemaType `json:"items"`
	Type  string          `json:"type"`
}

// JSONSchemaPrimitive represents a JSON schema primitive type.
type JSONSchemaPrimitive struct {
	Type        string `json:"type"`
	Description string `json:"description,omitempty"`
}

// NewJSONSchema creates a full JSON schema from a struct type.
// Struct field descriptions should be included within "doc:" tags: `json:"name" doc:"The person's full name"`
// Optional fields can be marked with `omitempty"`
func NewJSONSchema(name, description string, obj any) (*JSONSchema, error) {
	schema, err := NewJSONSchemaPartial(obj)
	if err != nil {
		return nil, fmt.Errorf("failed to create schema from struct: %w", err)
	}

	return &JSONSchema{
		Name:        name,
		Description: description,
		Schema:      &schema,
		Strict:      true,
	}, nil
}

// NewJSONSchemaPartial creates a partial JSON schema from a struct type.
// Struct field descriptions should be included within "doc:" tags: `json:"name" doc:"The person's full name"`
// Optional fields can be marked with `omitempty"`
func NewJSONSchemaPartial(input any) (JSONSchemaType, error) {
	t := reflect.TypeOf(input)

	// Handle nil input
	if t == nil {
		return nil, errors.New("input is nil")
	}

	// If it's a pointer, get the underlying type
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	// Only structs are supported
	if t.Kind() != reflect.Struct {
		return nil, fmt.Errorf("input must be a struct or pointer to struct, got %s", t.Kind())
	}

	schema := &JSONSchemaObject{
		Type:       "object",
		Properties: make(map[string]JSONSchemaType),
		Required:   []string{},
	}

	// Process each field in the struct
	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)

		// Skip unexported fields
		if !field.IsExported() {
			continue
		}

		// Get the JSON tag if it exists
		jsonTag := field.Tag.Get("json")
		if jsonTag == "-" {
			continue // Skip fields with json:"-"
		}

		fieldName := extractFieldName(jsonTag, field.Name)

		// Check if this field is required
		if !isFieldOptional(jsonTag) {
			schema.Required = append(schema.Required, fieldName)
		}

		// Get the description from the doc tag if it exists
		description := field.Tag.Get("doc")

		// Process the field type
		fieldSchema, err := typeToSchema(field.Type, description)
		if err != nil {
			return nil, fmt.Errorf("failed to process field %s: %w", field.Name, err)
		}

		schema.Properties[fieldName] = &fieldSchema
	}

	return schema, nil
}

// Helper function to extract the field name from JSON tag
func extractFieldName(jsonTag, defaultName string) string {
	if jsonTag == "" {
		return defaultName
	}

	parts := strings.Split(jsonTag, ",")
	if parts[0] != "" {
		return parts[0]
	}

	return defaultName
}

// Helper function to check if a field is optional based on JSON tag
func isFieldOptional(jsonTag string) bool {
	parts := strings.Split(jsonTag, ",")
	for _, part := range parts {
		if part == "omitempty" {
			return true
		}
	}
	return false
}

// Convert a reflect.Type to a JSONSchema
func typeToSchema(t reflect.Type, description string) (JSONSchemaType, error) {
	schema := JSONSchemaPrimitive{
		Description: description,
	}

	// Handle pointers
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
		// Pointers are implicitly optional, but we don't modify the schema here
		// since that's handled by the omitempty tag
	}

	// Handle different types
	switch t.Kind() {
	case reflect.Bool:
		schema.Type = "boolean"

	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		schema.Type = "integer"

	case reflect.Float32, reflect.Float64:
		schema.Type = "number"

	case reflect.String:
		schema.Type = "string"

	case reflect.Struct:
		// Recursively process struct
		return NewJSONSchemaPartial(reflect.New(t).Elem().Interface())

	case reflect.Slice, reflect.Array:
		// Recursively process arrays and slices
		itemSchema, err := typeToSchema(t.Elem(), "")
		if err != nil {
			return nil, err
		}
		return JSONSchemaArray{
			Type:  "array",
			Items: &itemSchema,
		}, nil

	default:
		return nil, fmt.Errorf("unsupported type: %s", t.Kind())
	}

	return schema, nil
}
