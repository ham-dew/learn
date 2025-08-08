package kmeans

import (
	"fmt"
	"math/rand"

	"learn/mat/utils"

	"gonum.org/v1/gonum/mat"
)

type Kmeans struct {
	Data            *mat.Dense
	Labels          []int
	Representatives *mat.Dense

	clusterNum int
	rand       *rand.Rand
	rows       int
}

func New(clusters int, rand *rand.Rand) *Kmeans {
	return &Kmeans{
		Data:       nil,
		clusterNum: clusters,
		rand:       rand,
	}
}

func (k *Kmeans) Fit(samples *mat.Dense) error {
	k.Data = samples
	k.rows = samples.RawMatrix().Rows

	// make Representatives
	k.Representatives = k.initRepresentatives()

	// init Labels

	return nil
}

func (k *Kmeans) initRepresentatives() (res *mat.Dense) {
	transData := mat.DenseCopyOf(k.Data.T())
	minMax := make([]map[string]float64, 0)

	for i := range k.rows {
		row := transData.RowView(i)
		min := mat.Min(row)
		max := mat.Max(row)

		minMax = append(minMax, map[string]float64{"min": min, "max": max})
	}

	rep := make([]float64, k.rows)
	for j, v := range minMax {
		rep[j] = k.rand.Float64()*(v["max"]-v["min"]) + v["min"]
	}

	res = mat.NewDense(k.clusterNum, k.rows, nil)
	for i := range k.clusterNum {
		res.SetRow(i, rep)
	}

	return
}

func (k *Kmeans) calcLabels() (newLabel []int, err error) {
	rp_row, _ := k.Representatives.Caps()
	for i := range k.rows {
		distances := make([]float64, 0)
		for j := range rp_row {
			distance, err := utils.EuclideanDistance(k.Data.RowView(i), k.Representatives.RowView(j))
			if err != nil {
				return newLabel, fmt.Errorf("calcLabels: failed to calc distance [%w]", err)
			}
			distances = append(distances, distance)
		}

		newLabel = append(newLabel, utils.MinIndex(distances))
	}

	return newLabel, nil
}
