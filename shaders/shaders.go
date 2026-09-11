// Package shaders embeds pre-compiled SPIR-V binaries.
//
// Regenerate with `go generate ./...` after editing any .comp source
// (requires glslc, part of the Vulkan SDK / shaderc).
package shaders

//go:generate glslc --target-env=vulkan1.2 -O -o double.spv double.comp

import _ "embed"

//go:embed double.spv
var Double []byte
