package rbf;

public class HiddenLayer
{
    private double[] weights;

    private double[] exampleValues;

    public HiddenLayer(double[] values)
    {
        weights = new double[values.length];
        // randomly create all of the weights
        for (int i = 0; i < values.length; i++)
        {
            weights[i] =  (Math.random() * 100);
        }
        this.exampleValues = values;
    }

    public double activationFunction(double[] inputs)
    {
        double totalValue = 0;
        for (int i = 0; i < inputs.length; i++)
        {
            totalValue += getResult(inputs[i], this.weights[i], this.exampleValues[i]);
        }

        return totalValue;
    }

    private double getResult(double input, double weight, double example)
    {
        double result = 0;

        result = input - example;

        result *= result;

        result *= weight;

        return result;
    }
}