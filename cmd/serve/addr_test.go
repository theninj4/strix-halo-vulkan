package main

import "testing"

func TestOpenToNetwork(t *testing.T) {
	for _, tt := range []struct {
		addr string
		open bool
	}{
		{"0.0.0.0:11434", true},
		{":11434", true},
		{"127.0.0.1:8080", false},
		{"localhost:8080", false},
		{"[::1]:8080", false},
		{"192.168.1.10:11434", true},
		{"[::]:11434", true},
		{"garbage", true},
	} {
		if got := openToNetwork(tt.addr); got != tt.open {
			t.Errorf("openToNetwork(%q) = %v, want %v", tt.addr, got, tt.open)
		}
	}
}
