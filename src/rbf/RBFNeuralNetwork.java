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
                inputs[j] = Math.random() * 10 - 5;
            }

            double y = findCorrectAnswer(inputs);
            this.hiddenNodes[i] = new HiddenLayer(inputs, y);
        }
    }

    /**
     * this method will create hidden nodes from centroids with the number of inputs specified
     * by size
     */
    public void createTrainingDataWithKClustering(int centroids, double[][] buildingSet, int n)
    {
        //double[][] inputs = new double[size][n];
        this.n = n;

//        for (int i = 0; i < buildingSet.length; i++)
//        {
//            double[] input = new double[n];
//            for (int j = 0; j < n; j++)
//            {
//                input[j] = Math.random() * 10 - 5;
//               // System.out.print(input[j] + ", ");
//            }
//
//            inputs[i] = input;
//           // System.out.println("");
//        }

        kMeansClustering clusterer = new kMeansClustering(buildingSet);
        double[][] createdCentroids = clusterer.run(5.0, centroids);
        this.hiddenNodes = new HiddenLayer[centroids];
        this.nodeWeights = new double[centroids];

        for (int k = 0; k < createdCentroids.length; k++)
        {
            double y = findCorrectAnswer(createdCentroids[k]);
            this.hiddenNodes[k] = new HiddenLayer(createdCentroids[k], y);
        }

        for (int l = 0; l < centroids; l++)
        {
            this.nodeWeights[l] = Math.random();
            System.out.println("Weights: " + this.nodeWeights[l]);
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

    public double getResult(double[] inputs)
    {
        double result = 0.0;

        for (int i = 0; i < this.hiddenNodes.length; i++)
        {
            result += (this.hiddenNodes[i].activationFunction(inputs) * this.nodeWeights[i]);
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

    /**
     * This method is for training the data
     */
    public void train(double[][] trainingSet)
    {
        for (int i = 0; i < trainingSet.length; i++)
        {
//            double[] input = new double[this.n];
//            for (int j = 0; j < this.n; j++)
//            {
//                input[j] = Math.random() * 10 - 5;
//            }

            double correctAnswer = findCorrectAnswer(trainingSet[i]);
            double rbfAnswer = getResult(trainingSet[i]);
            double error = correctAnswer - rbfAnswer;
            backProp(trainingSet[i], error, rbfAnswer);
        }
    }

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

    public void updateNodeWeights(double error)
    {
        for (int i = 0; i < this.nodeWeights.length; i++)
        {
            this.nodeWeights[i] += ada * error / this.nodeWeights[i];
        }
    }
}