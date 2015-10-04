package rbf;

public class HiddenLayer
{
    public double[] weights;
    public double[] exampleValues;
    public double result;
    public double sigma = 0.5;

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
        for (int i = 0; i < inputs.length; i++)
        {
            totalValue += calculateInputValue(inputs, i);
        }

        totalValue = totalValue * this.result;

        return totalValue;
    }

    public double calculateInputValue(double[] input, int index)
    {
        // exp( -1/2sigma * (|| xi - xj || ) ^ 2) * wi
        return this.weights[index] * Math.exp(-1 * (0.5 / this.sigma) * (input[index] - this.exampleValues[index]) * (input[index] - this.exampleValues[index]));
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