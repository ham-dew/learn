package kmeans

import (
	"fmt"
	"math/rand"
	"reflect"
	"slices"

	"learn/slice/utils"
)

type Kmeans struct {
	Data            [][]float64
	Labels          []int
	Representatives [][]float64
	BestIter        int

	max_iter   int
	clusterNum int
	rand       *rand.Rand
}

func New(clusters, max_iter int, rand *rand.Rand) *Kmeans {
	if max_iter < 1 {
		max_iter = 100
	}
	return &Kmeans{
		clusterNum: clusters,
		max_iter:   max_iter,
		rand:       rand,
	}
}

func (k *Kmeans) Fit(samples [][]float64) error {
	k.Data = samples

	// make Representatives
	k.Representatives = k.initRepresentatives()

	// make Labels
	// init Labels
	initLabels, err := k.calcLabels()
	if err != nil {
		return fmt.Errorf("kmeans Fit: failed to calc init label [%w]", err)
	}
	k.Labels = make([]int, len(initLabels))
	copy(k.Labels, initLabels)

	// update data
	i := 1
	for i = 1; i < k.max_iter; i++ {
		// update Representatives
		k.Representatives = k.calcRepresentatives()
		// update Labels
		newLabel, err := k.calcLabels()
		if err != nil {
			k.BestIter = i
			return fmt.Errorf("kmeans Fit: failed to calc label [%w]", err)
		}

		// check representative
		if reflect.DeepEqual(k.Labels, newLabel) {
			break
		}
		k.Labels = newLabel
	}
	k.BestIter = i

	return nil
}

func (k *Kmeans) initRepresentatives() (res [][]float64) {
	transData := utils.Transpose(k.Data)
	minMax := make([]map[string]float64, 0)
	for _, d := range transData {
		min := slices.Min(d)
		max := slices.Max(d)

		minMax = append(minMax, map[string]float64{"min": min, "max": max})
	}

	for i := range k.clusterNum {
		res = append(res, make([]float64, len(minMax)))
		for j, v := range minMax {
			res[i][j] = k.rand.Float64()*(v["max"]-v["min"]) + v["min"]
		}
	}

	return res
}

func (k *Kmeans) calcRepresentatives() (newRepresentatives [][]float64) {
	for i := range k.clusterNum {
		var grouped [][]float64
		for j, d := range k.Data {
			if k.Labels[j] == i {
				grouped = append(grouped, d)
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
			newRepresentatives = append(newRepresentatives, updated)
		}
	}
	return newRepresentatives
}

func (k *Kmeans) calcLabels() (newLabel []int, err error) {
	for _, d := range k.Data {
		distances := make([]float64, 0)
		for _, r := range k.Representatives {
			distance, err := utils.EuclideanDistance(d, r)
			if err != nil {
				return newLabel, fmt.Errorf("calcLabels: failed to calc distance [%w]", err)
			}
			distances = append(distances, distance)
		}
		newLabel = append(newLabel, utils.MinIndex(distances))
	}
	return newLabel, nil
}
