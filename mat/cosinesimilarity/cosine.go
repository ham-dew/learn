package cosinesimilarity

import (
	"fmt"
	"math"

	"learn/utils"

	"gonum.org/v1/gonum/mat"
)

// similarity = <X, Y> / (||X||*||Y||)
func Compute(vectorX, vectorY *mat.Dense) (float64, error) {
	// dot product
	dotmat, err := utils.Dot(vectorX, vectorY)
	if err != nil {
		return 0, fmt.Errorf("CosineDistance: failed dot x,y [%w]", err)
	}
	dotXY := dotmat.At(0, 0)

	// normalization
	dotmat, err = utils.Dot(vectorX, vectorX) // sum(x*x)
	if err != nil {
		return 0, fmt.Errorf("CosineDistance: failed dot x,x [%w]", err)
	}
	normX := math.Sqrt(dotmat.At(0, 0))

	dotmat, err = utils.Dot(vectorY, vectorY) // sum(y*y)
	if err != nil {
		return 0, fmt.Errorf("CosineDistance: failed dot y,y [%w]", err)
	}
	normY := math.Sqrt(dotmat.At(0, 0))

	return dotXY / (normX * normY), nil
}
