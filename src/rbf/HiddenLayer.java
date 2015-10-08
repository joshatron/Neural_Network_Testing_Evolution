package rbf;

public class HiddenLayer
{
    public double[] weights;
    public double[] exampleValues;
    public double result;
    public double sigma = 0.5;
    private int n;
    int basisFunc = 0;

    public HiddenLayer(double[] values, double result, int n)
    {
        weights = new double[values.length];
        this.n = n;
        // randomly create all of the weights
        for (int i = 0; i < this.n; i++)
        {
            weights[i] =  Math.random();
        }
        this.exampleValues = values;
        this.result = result;
    }

    public double activationFunction(double[] inputs)
    {
        double totalValue = 0;
        for (int i = 0; i < this.n; i++)
        {
            totalValue += calculateInputValue(inputs, i);
        }

        totalValue = totalValue * this.result;

        return totalValue;
    }

    public double calculateInputValue(double[] input, int index)
    {
        if (this.basisFunc == 0)
        {
            // exp( -1/2sigma * (|| xi - xj || ) ^ 2) * wi
            return this.weights[index] * Math.exp(-1 * (0.5 / this.sigma) * Math.abs((input[index] - this.exampleValues[index]) * (input[index] - this.exampleValues[index])));
        }
        else if (this.basisFunc == 1)
        {
            // linear basis function
            // w * (1 / (|xi - xj|))
            return (1 / (Math.abs(input[index] - this.exampleValues[index]))) * this.weights[index];
        }
        else
        {
            return 0.0;
        }
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