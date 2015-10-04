package rbf;

public class HiddenLayer
{
    private double[] weights;
    private double[] exampleValues;
    private double result;
    private double sigma = 0.5;

    public HiddenLayer(double[] values, double result)
    {
        weights = new double[values.length];
        // randomly create all of the weights
        for (int i = 0; i < values.length; i++)
        {
            weights[i] =  (Math.random() * 100);
        }
        this.exampleValues = values;
        this.result = result;
    }

    public double activationFunction(double[] inputs)
    {
        double totalValue = 0;
        double totalWeights = 0;
        for (int i = 0; i < inputs.length; i++)
        {
            // exp( -1/2sigma * (|| xi - xj || ) ^ 2) * wi
            totalValue += this.weights[i] * Math.exp(-1 * (0.5 / this.sigma) * (inputs[i] - this.exampleValues[i]) * (inputs[i] - this.exampleValues[i]));
            totalWeights += this.weights[i];
        }

        totalValue /= totalWeights;

        totalValue = totalValue * this.result;

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

    public void print()
    {
        for (int i = 0; i < this.exampleValues.length; i++)
        {
            System.out.print("Input["+i+"] = " + this.exampleValues[i] + ", ");
        }
        System.out.println("");
    }
}