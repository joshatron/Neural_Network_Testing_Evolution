package rbf;

public class RBFNeuralNetwork
{
    HiddenLayer[] hiddenNodes;

    /**
     * this method creates 
     */
    public void createTrainingData(int size, int n)
    {
        this.hiddenNodes = new HiddenLayer[size];

        for (int i = 0; i < size; i++)
        {
            double[] inputs = new double[n];
            for (int j = 0; j < n; j++)
            {
                inputs[j] = Math.random() * 100;
            }

            double y = findCorrectAnswer(inputs);
            this.hiddenNodes[i] = new HiddenLayer(inputs, y);
        }
    }

    /**
     * this method will create hidden nodes from centroids with the number of inputs specified
     * by size
     */
    public void createTrainingDataWithKClustering(int centroids, int size, int n)
    {
        double[][] inputs = new double[size][n];

        for (int i = 0; i < size; i++)
        {
            double[] input = new double[n];
            for (int j = 0; j < n; j++)
            {
                input[j] = Math.random() * 100;
            }

            inputs[i] = input;
        }

        kMeansClustering clusterer = new kMeansClustering(inputs);
        double[][] createdCentroids = clusterer.run(5.0, centroids);
        this.hiddenNodes = new HiddenLayer[centroids];

        for (int k = 0; k < createdCentroids.length; k++)
        {

        }
    }

    public double getResult(double[] inputs)
    {
        double result = 0.0;

        for (int i = 0; i < this.hiddenNodes.length; i++)
        {
            result += this.hiddenNodes[i].activationFunction(inputs);
        }

        return result;
    }


    public double findCorrectAnswer(double[] inputs)
    {
        double result = 0;
        for (int i = 0; i < inputs.length - 1; i++)
        {
            result += (1 - inputs[i])*(1 - inputs[i]) + (100 * (inputs[i+1] - inputs[i] * inputs[i]) * (inputs[i+1] - inputs[i] * inputs[i]));
        }
        return result;
    }
}