package feedforward;

/**
 * Created by joshua on 10/4/15.
 */
public class RunFeedForward
{
    public static FeedForwardNeuralNetwork run(double[][] examples, int hiddenLayers, int[] sizesLayers,
                                               double learningRate, double momentum,
                                               ActivationFunction activationFunction)
    {
        FeedForwardNeuralNetwork net = new FeedForwardNeuralNetwork(hiddenLayers, sizesLayers, activationFunction,
                                                                    momentum, learningRate);
        for(int k = 0; k < examples.length; k++)
        {
            if(k % 1000 == 0)
            {
                System.out.println(((k / examples.length) * 100) + "% done");
            }
            double[] inputs = new double[examples[0].length - 1];
            double[] output = new double[1];

            for(int a = 0; a < inputs.length; a++)
            {
                inputs[a] = examples[k][a];
            }
            output[0] = examples[k][examples[0].length - 1];

            net.backprop(inputs, output);

        }

        return net;
    }
}
