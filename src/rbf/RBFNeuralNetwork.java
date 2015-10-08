package rbf;

public class RBFNeuralNetwork
{
    HiddenLayer[] hiddenNodes;
    double[] nodeWeights;
    int n;
    double ada = 0.05;

    public RBFNeuralNetwork(int n, double ada)
    {
        this.n = n;
        this.ada = ada;
    }

    /**
     * this method creates random training data without kClustering
     */
    public void createTrainingData(int size, int n)
    {
        this.hiddenNodes = new HiddenLayer[size];

        for (int i = 0; i < size; i++)
        {
            double[] inputs = new double[n];
            for (int j = 0; j < n; j++)
            {
                inputs[j] = Math.random() * 10 - 5;
            }

            double y = findCorrectAnswer(inputs);
            this.hiddenNodes[i] = new HiddenLayer(inputs, y, n);
        }
    }

    /**
     * this method will create hidden nodes from centroids with the number of inputs specified
     * by size using k-means-Clustering
     */
    public void createTrainingDataWithKClustering(int centroids, double[][] buildingSet)
    {
        double[][] data;
        if (buildingSet[0].length > this.n)
        {
            System.out.println("redoing data");
            data = new double[buildingSet.length][this.n];

            for (int i = 0; i < buildingSet.length; i++)
            {
                for (int j = 0; j < this.n; j++)
                {
                    data[i][j] = buildingSet[i][j];
                }
            }
        }
        else
        {
            data = buildingSet;
        }
        kMeansClustering clusterer = new kMeansClustering(data, this.n);
        double[][] createdCentroids = clusterer.run(5.0, centroids);
        this.hiddenNodes = new HiddenLayer[centroids];
        this.nodeWeights = new double[centroids];

        for (int k = 0; k < createdCentroids.length; k++)
        {
            double y = findCorrectAnswer(createdCentroids[k]);
            this.hiddenNodes[k] = new HiddenLayer(createdCentroids[k], y, this.n);
        }

        for (int l = 0; l < centroids; l++)
        {
            this.nodeWeights[l] = Math.random();
        }
    }

    /**
     * this method will create hidden nodes from centroids with the number of inputs specified
     * by size
     */
    public void createRandomTrainingDataWithKClustering(int centroids, int size)
    {
        double[][] inputs = new double[size][this.n];

        for (int i = 0; i < size; i++)
        {
            double[] input = new double[this.n];
            for (int j = 0; j < this.n; j++)
            {
                input[j] = Math.random() * 10 - 5;
            }

            inputs[i] = input;
        }

        kMeansClustering clusterer = new kMeansClustering(inputs, this.n);
        double[][] createdCentroids = clusterer.run(5.0, centroids);
        this.hiddenNodes = new HiddenLayer[centroids];
        this.nodeWeights = new double[centroids];

        for (int k = 0; k < createdCentroids.length; k++)
        {
            double y = findCorrectAnswer(createdCentroids[k]);
            this.hiddenNodes[k] = new HiddenLayer(createdCentroids[k], y, this.n);
        }

        for (int l = 0; l < centroids; l++)
        {
            this.nodeWeights[l] = Math.random();
        }
    }

    /**
     * This method prints out the hidden nodes for debugging purposes
     */
    public void print()
    {
        for (int i = 0; i < this.hiddenNodes.length; i++)
        {
            this.hiddenNodes[i].print();
        }
    }

    /**
     * This method runs the inputs through the rbf network and
     * returns the result
     */
    public double getResult(double[] inputs)
    {
        double result = 0.0;

        for (int i = 0; i < this.hiddenNodes.length; i++)
        {
//            if (inputs.length > this.n)
//            {
//                double[] tempInputs = new double[inputs.length - 1];
//
//                for (int j = 0; j < tempInputs.length; j++)
//                {
//                    tempInputs[j] = inputs[j];
//                }
//                result += (this.hiddenNodes[i].activationFunction(tempInputs) * this.nodeWeights[i]);
//            }
//            else
//            {
                result += (this.hiddenNodes[i].activationFunction(inputs) * this.nodeWeights[i]);
//            }
        }

        return result;
    }


    /**
     * This method finds the exact value of the function
     */
    public double findCorrectAnswer(double[] inputs)
    {
        double result = 0;
        for (int i = 0; i < this.n - 1; i++)
        {
            result += (1 - inputs[i])*(1 - inputs[i]) + (100 * (inputs[i+1] - inputs[i] * inputs[i]) * (inputs[i+1] - inputs[i] * inputs[i]));
        }
        return result;
    }

    /**
     * This method is for training the data
     */
    public void train(double[][] trainingSet)
    {
        for (int i = 0; i < trainingSet.length; i++)
        {
            double correctAnswer = this.findCorrectAnswer(trainingSet[i]);
            double rbfAnswer = this.getResult(trainingSet[i]);
            double error = correctAnswer - rbfAnswer;
            System.out.println("Error: " + error);
            this.backProp(trainingSet[i], error, rbfAnswer);
        }
    }

    /**
     * This method is for training the data
     */
    public void trainRandom(int size)
    {
        for (int i = 0; i < size; i++)
        {
            double[] input = new double[this.n];
            for (int j = 0; j < this.n; j++)
            {
                input[j] = Math.random() * 10 - 5;
            }

            double correctAnswer = findCorrectAnswer(input);
            double rbfAnswer = getResult(input);
            double error = correctAnswer - rbfAnswer;
            backProp(input, error, rbfAnswer);
        }
    }

    /**
     * This method takes the error and updates the weight using gradient descent
     * to
     */
    public void backProp(double[] input, double error, double total)
    {
        for (int i = 0; i < this.hiddenNodes.length; i++)
        {
            double nodeOutput = this.hiddenNodes[i].activationFunction(input);
            double errorValue = error * nodeOutput * this.nodeWeights[i] / total;
            double errorValue2 = Math.exp(-1 / Math.abs(errorValue));
            //System.out.println("Error: " + error + ", change: " + errorValue + ", to e: " + errorValue2);
            if (error < 0)
            {
                this.nodeWeights[i] = this.nodeWeights[i] - this.nodeWeights[i] * errorValue2 * this.ada;
            }
            else if (error > 0)
            {
                this.nodeWeights[i] = this.nodeWeights[i] + this.nodeWeights[i] * errorValue2 * this.ada;
            }

            // use error for node to update error for each weight
            for (int j = 0; j < this.hiddenNodes[i].weights.length; j++)
            {
                double innerWeightChange = errorValue2 * this.hiddenNodes[i].calculateInputValue(input, j) * this.ada;

                if (error < 0)
                {
                    this.hiddenNodes[i].weights[j] -= innerWeightChange;
                }
                else if (error > 0)
                {
                    this.hiddenNodes[i].weights[j] += innerWeightChange;
                }
            }
        }
    }
}