package rbf;

public class HiddenLayer
{
    public double[] weights;
    public double[] exampleValues;
    public double result;
    public double sigma = 0.5;
    private int n;
    int basisFunc = 0;

    public HiddenLayer(double[] values, double result, int n, int basisFunc)
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
        this.basisFunc = basisFunc;
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
        // Gausian basis function
        if (this.basisFunc == 0)
        {
            // exp( -1/2sigma * (|| xi - xj || ) ^ 2) * wi
            return this.weights[index] * Math.exp(-1 * (0.5 / this.sigma) * Math.abs((input[index] - this.exampleValues[index]) * (input[index] - this.exampleValues[index])));
        }
        // Inverse Multi-Quadric Function
        else if (this.basisFunc == 1)
        {
            double err = Math.abs(input[index] - this.exampleValues[index]);
            // ( || xi - xj || ^  2 + sigma ^ 2 ) ^ (-1/2)
            return Math.pow((err * err + this.sigma * this.sigma), -0.5);
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