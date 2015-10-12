package rbf;

public class kMeansClustering
{
    private double[][] inputs;
    private int n;

    public kMeansClustering(double[][] inputs, int n)
    {
        this.inputs = inputs;
        this.n = n;
    }

    /**
     * This method randomly picks centroids from the inputs
     *
     * @param numbOfCentroids
     * @return
     */
    public double[][] createInitialCentroids(int numbOfCentroids)
    {
        if (this.inputs.length <= numbOfCentroids)
        {
            return this.inputs;
        }

        double[][] centroids = new double[numbOfCentroids][n];

        int[] assignedCentroids = new int[numbOfCentroids];
        int numbOfAssignedCentroids = 0;

        for (int i = 0; i < numbOfCentroids; i++)
        {
            assignedCentroids[i] = -1;
        }

        while (numbOfAssignedCentroids < numbOfCentroids)
        {
            int nextCentroid = (int) (Math.random() * (this.inputs.length - 1));

            if (!findIfValueIsInArray(assignedCentroids,nextCentroid))
            {
                centroids[numbOfAssignedCentroids] = inputs[nextCentroid];
                assignedCentroids[numbOfAssignedCentroids] = nextCentroid;
                numbOfAssignedCentroids++;

            }
        }

        return centroids;
    }

    public boolean findIfValueIsInArray(int[] array, int value)
    {
        for (int i = 0; i < array.length; i++)
        {
            if (array[i] == value)
            {
                return true;
            }
        }

        return false;
    }

    /**
     * This method will converge the centroids until the max distance
     * changed is less than the param maxDiff
     *
     * @param maxDiff
     * @return
     */
    public double[][] run(double maxDiff, int numbOfCentroids)
    {
        double[][] centroids = createInitialCentroids(numbOfCentroids);

        double diff = 0;

        // run till convergence
        do {
            // assign all inputs to nearest centroid
            double[][][] inputsWithCentroids = assignInputsToCentroids(centroids);

            double[][] newCentroids = new double[centroids.length][];
            // calculate new centroids based on "groups"
            for (int i = 0; i < centroids.length; i++)
            {
                newCentroids[i] = calcNewCentroids(inputsWithCentroids[i]);
            }

            // calculate change in centroids
            diff = maxCentroidChange(centroids, newCentroids);
            centroids = newCentroids;
        } while (diff > maxDiff);

        return centroids;
    }

    /**
     * This method finds the new centroids
     *
     * @param centroidInputs
     * @return
     */
    public double[] calcNewCentroids(double[][] centroidInputs)
    {
        double[] centroid = new double[centroidInputs[0].length];

        // find sum for all inputs
        for (int i = 0; i < centroidInputs.length; i++)
        {
            for (int j = 0; j < n; j++)
            {
                centroid[j] += centroidInputs[i][j];
            }
        }

        // find average with sum and # of values
        for (int k = 0; k < centroid.length; k++)
        {
            centroid[k] /= centroidInputs.length;
        }

        return centroid;
    }

    /**
     * This method assigns all inputs to the nearest centroid
     */
    public double[][][] assignInputsToCentroids(double[][] centroids)
    {
        int[] inputMapToCentroids = new int[centroids.length];
        int[] indexInputCentroids = new int[this.inputs.length];

        for (int b = 0; b < indexInputCentroids.length; b++)
        {
            indexInputCentroids[b] = 0;
        }

        for (int i = 0; i < this.inputs.length; i++)
        {
            double minDist = 999999;
            int index = 0;
            for (int j = 0; j < centroids.length; j++)
            {
                double currentDist = distance(centroids[j], this.inputs[i]);
                if (currentDist < minDist)
                {
                    minDist = currentDist;
                    index = j;
                }
            }

            indexInputCentroids[i] = index;
            inputMapToCentroids[index]++;
        }

        double[][][] inputsInCentroids = new double[centroids.length][][];
        for (int k = 0; k < inputMapToCentroids.length; k++)
        {
            inputsInCentroids[k] = new double[inputMapToCentroids[k] + 1][this.inputs[0].length];
        }

        int[] currentIndexForCentroids = new int[centroids.length];

        for (int a = 0; a < centroids.length; a++)
        {
            inputsInCentroids[a][0] = centroids[a];
            currentIndexForCentroids[a] = 1;
        }

        for (int l = 0; l < indexInputCentroids.length; l++)
        {
            int index = indexInputCentroids[l];
            inputsInCentroids[index][currentIndexForCentroids[index]] = this.inputs[l];
            currentIndexForCentroids[index]++;
        }

        return inputsInCentroids;
    }

    /**
     * This method calculates a Euclidean distance between two points
     */
    public double distance(double[] x1, double[] x2)
    {
        double total = 0.0;

        for (int i = 0; i < n; i++)
        {
            total += (x1[i] - x2[i]) * (x1[i] - x2[i]);
        }

        return Math.sqrt(total);
    }


    /**
     * This method calculates the maximum change in centroid location
     */
    public double maxCentroidChange(double[][] oldCentroids, double[][] newCentroids)
    {
        double maxDist = 0;

        for (int i = 0; i < oldCentroids.length; i++)
        {
            double currentDist = distance(oldCentroids[i], newCentroids[i]);

            if (currentDist > maxDist)
            {
                maxDist = currentDist;
            }
        }

        return maxDist;
    }
}
