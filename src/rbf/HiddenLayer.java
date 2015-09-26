package rbf;

public class HiddenLayer
{
    private int[] weights;

    private int[] exampleValues;

    public HiddenLayer(int[] values)
    {
        // randomly create all of the weights
        for (int i = 0; i < values.length; i++)
        {
            weights[i] = (int) (Math.random() * 100);
        }
        this.exampleValues = values;
    }

    public int activationFunction(int[] inputs)
    {
        int totalValue = 0;
        for (int i = 0; i < inputs.length; i++)
        {
            totalValue += getResult(inputs[i], this.weights[i], this.exampleValues[i]);
        }

        return totalValue;
    }

    public int getResult(int input, int weight, int example)
    {
        int result = 0;

        result = input - example;

        result *= result;

        result *= weight;

        return result;
    }
}