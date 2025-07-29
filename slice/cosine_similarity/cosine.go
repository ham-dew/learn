package cosinesimilarity

import (
	"fmt"
	"math"

	"learn/slice/utils"
)

// similarity = <X, Y> / (||X||*||Y||)
func Compute(vectorX, vectorY []float64) (float64, error) {
	// dot product
	dotXY, err := utils.Dot(vectorX, vectorY)
	if err != nil {
		return 0, fmt.Errorf("CosineDistance: failed dot x,y [%w]", err)
	}

	// normalization
	d, err := utils.Dot(vectorX, vectorX) // sum(x*x)
	if err != nil {
		return 0, fmt.Errorf("CosineDistance: failed dot x,x [%w]", err)
	}
	normX := math.Sqrt(d)

	d, err = utils.Dot(vectorY, vectorY) // sum(y*y)
	if err != nil {
		return 0, fmt.Errorf("CosineDistance: failed dot y,y [%w]", err)
	}
	normY := math.Sqrt(d)

	return dotXY / (normX * normY), nil
}
