package rbf;

public class RBFNeuralNetwork
{
    HiddenLayer[] hiddenNodes;

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
            this.hiddenNodes[i] = new HiddenLayer(inputs);
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