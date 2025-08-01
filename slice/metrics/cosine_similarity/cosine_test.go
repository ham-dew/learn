package cosinesimilarity

import "testing"

func TestCompute(t *testing.T) {
	type args struct {
		vectorX []float64
		vectorY []float64
	}
	tests := []struct {
		name    string
		args    args
		want    float64
		wantErr bool
	}{
		{
			name: "pass1",
			args: args{
				vectorX: []float64{3, 2, 0, 5},
				vectorY: []float64{1, 0, 0, 0},
			},
			want:    0.48666426339228763,
			wantErr: false,
		},
		{
			name: "pass2",
			args: args{
				vectorX: []float64{4, 5, 2},
				vectorY: []float64{4, 4, 3},
			},
			want:    0.9778024140774095,
			wantErr: false,
		},
		{
			name: "fail",
			args: args{
				vectorX: []float64{4, 5, 2, 3},
				vectorY: []float64{4, 4, 3},
			},
			want:    0.0,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Compute(tt.args.vectorX, tt.args.vectorY)
			if (err != nil) != tt.wantErr {
				t.Errorf("Compute() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("Compute() = %v, want %v", got, tt.want)
			}
		})
	}
}
