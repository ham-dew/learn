package kmeans

import (
	"fmt"
	"math/rand"
	"reflect"

	"learn/mat/utils"

	"gonum.org/v1/gonum/mat"
)

type Kmeans struct {
	Data            *mat.Dense
	Labels          []int
	Representatives *mat.Dense
	BestIter        int

	max_iter    int
	cluster_num int
	rand        *rand.Rand
	rows        int
	cols        int
}

func New(clusters, max_iter int, rand *rand.Rand) *Kmeans {
	if max_iter < 1 {
		max_iter = 100
	}
	return &Kmeans{
		Data:        nil,
		cluster_num: clusters,
		max_iter:    max_iter,
		rand:        rand,
	}
}

func (k *Kmeans) Fit(samples *mat.Dense) error {
	k.Data = samples
	k.rows = samples.RawMatrix().Rows
	k.cols = samples.RawMatrix().Cols

	// make Representatives
	k.Representatives = k.initRepresentatives()

	// init Labels
	initLabels, err := k.calcLabels()
	if err != nil {
		return fmt.Errorf("kmeans Fit: failed to calc init label [%w]", err)
	}
	k.Labels = make([]int, len(initLabels))
	copy(k.Labels, initLabels)

	for iter := range k.max_iter {
		// update Representatives
		k.Representatives = k.calcRepresentatives()
		// update Labels
		newLabel, err := k.calcLabels()
		if err != nil {
			return fmt.Errorf("kmeans Fit: failed to calc label [%w]", err)
		}

		// check representative
		if reflect.DeepEqual(k.Labels, newLabel) {
			k.BestIter = iter
			return nil
		}
		k.Labels = newLabel
	}

	k.BestIter = k.max_iter
	return nil
}

func (k *Kmeans) initRepresentatives() (res *mat.Dense) {
	transData := mat.DenseCopyOf(k.Data.T())
	tr, _ := transData.Caps()
	minMax := make([]map[string]float64, tr)

	for i := range tr {
		row := transData.RowView(i)
		min := mat.Min(row)
		max := mat.Max(row)

		minMax[i] = map[string]float64{"min": min, "max": max}
	}

	res = mat.NewDense(k.cluster_num, tr, nil)
	for i := range k.cluster_num {
		rep := make([]float64, tr)
		for j, v := range minMax {
			rep[j] = k.rand.Float64()*(v["max"]-v["min"]) + v["min"]
		}
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
		fmt.Println("check distances = ", distances)
		newLabel = append(newLabel, utils.MinIndex(distances))
	}
	return newLabel, nil
}

func (k *Kmeans) calcRepresentatives() (res *mat.Dense) {
	var newRepresentatives []float64
	for i := range k.cluster_num {
		var grouped [][]float64
		for j := range k.rows {
			if k.Labels[j] == i {
				row := k.Data.RawRowView(j)
				grouped = append(grouped, row)
			}
		}
		if len(grouped) != 0 {
			transposedGroup := utils.Transpose(grouped)
			updated := make([]float64, 0)
			for _, vectors := range transposedGroup {
				value := 0.0
				for _, v := range vectors {
					value += v
				}
				updated = append(updated, value/float64(len(vectors)))
			}
			newRepresentatives = append(newRepresentatives, updated...)
		}
	}
	res = mat.NewDense(k.cluster_num, k.cols, newRepresentatives)
	return
}
