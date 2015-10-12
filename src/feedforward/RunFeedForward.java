package feedforward;

import nettest.Trainer;

import java.io.File;
import java.io.IOException;

/**
 * Created by joshua on 10/4/15.
 */
public class RunFeedForward
{
    /**
     * Defines a neural network and trains it on an array of examples.
     * @param examples array containing examples, last 2 are expected outputs
     * @param hiddenLayers number of hidden layers
     * @param sizesLayers array containing sizes of each layer
     * @param learningRate learning rate for the net
     * @param momentum momentum for the net
     * @param activationFunction activation function to use
     * @param timesThrough number of times to run through the examples
     * @return the final net that was created
     */
    public static FeedForwardNeuralNetwork run(double[][] examples, int hiddenLayers, int[] sizesLayers,
                                               double learningRate, double momentum,
                                               ActivationFunction activationFunction, int timesThrough)
    {
        FeedForwardNeuralNetwork net = new FeedForwardNeuralNetwork(hiddenLayers, sizesLayers, activationFunction,
                                                                    momentum, learningRate);
        /*try
        {
            net.export(new File("start.json"));
        } catch(IOException e)
        {
            e.printStackTrace();
        }*/
        System.out.println("Running training");
        long time = System.currentTimeMillis();
        /*System.out.println(hiddenLayers);
        System.out.println(sizesLayers[0]);*/
        //run through the data set however many times
        for(int t = 0; t < timesThrough; t++)
        {
            //System.out.println(sizesLayers.length);
            //for each examples
            for(int k = 0; k < examples.length; k++)
            {
                //extract input and output
                if(k % 1000 == 0)
                {
                    //System.out.println(Math.round(((k / examples.length) * 100)) + "% done");
                }
                double[] input = new double[examples[0].length];
                double[] output = new double[2];

                for(int a = 0; a < input.length - 1; a++)
                {
                    input[a] = examples[k][a];
                }

                double tempOut = Trainer.plusOrMinus10(examples[k][input.length - 1]);
                input[input.length - 1] = tempOut;
                if(tempOut > examples[k][input.length - 1])
                {
                    output[0] = .9;
                    output[1] = .1;
                }
                else
                {
                    output[0] = .1;
                    output[1] = .9;
                }
                //run backprop
                net.backprop(input, output);

            }
        }

        /*try
        {
            net.export(new File("end" + examples.length + Math.random() + ".json"));
        } catch(IOException e)
        {
            e.printStackTrace();
        }*/
        System.out.println("Execution time: " + (System.currentTimeMillis() - time));
        return net;
    }
}
