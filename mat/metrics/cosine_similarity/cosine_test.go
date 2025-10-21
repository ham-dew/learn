package cosinesimilarity

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestCompute(t *testing.T) {
	type args struct {
		vectorX *mat.Dense
		vectorY *mat.Dense
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
				vectorX: mat.NewDense(1, 4, []float64{3, 2, 0, 5}),
				vectorY: mat.NewDense(1, 4, []float64{1, 0, 0, 0}),
			},
			want:    0.48666426339228763,
			wantErr: false,
		},
		{
			name: "pass2",
			args: args{
				vectorX: mat.NewDense(1, 3, []float64{4, 5, 2}),
				vectorY: mat.NewDense(1, 3, []float64{4, 4, 3}),
			},
			want:    0.9778024140774095,
			wantErr: false,
		},
		{
			name: "fail",
			args: args{
				vectorX: mat.NewDense(1, 4, []float64{4, 5, 2, 3}),
				vectorY: mat.NewDense(1, 3, []float64{4, 4, 3}),
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
