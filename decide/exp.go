package decide

// The readout's exp is the C library's, not Go's.
//
// v1 promises answers bit for bit, and both of its other implementations
// call glibc's exp: surogate's `std::exp` and the golden file's Python
// `math.exp`. Go's math.Exp is not correctly rounded and differs from glibc
// in the last ulp on about one argument in fifty, which moved five of the
// golden file's answers (R1).

// #cgo LDFLAGS: -lm
// #include <math.h>
import "C"

func exp(x float64) float64 { return float64(C.exp(C.double(x))) }
