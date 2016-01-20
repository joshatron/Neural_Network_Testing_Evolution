package feedforward;

/**
 * inputs: 5
 * middle: 10
 * output: 2
 */
public class FeedForwardNeuralNetwork
{
    private double[] weights;

    public FeedForwardNeuralNetwork(double[] weights)
    {
        this.weights = weights;
    }

    public void setWeights(double[] weights)
    {
        this.weights = weights;
    }

    public double[] compute(double[] input)
    {
        double[] weights = this.weights;
        double[] middle = {0,0,0,0,0,0,0,0,0,0};
        double[] output = {0,0};

        middle[0] = input[0] * weights[0] + input[1] * weights[10] + input[2] * weights[20] + input[3] * weights[30] + input[4] * weights[40] ;
        middle[0] = 1.0 / (1.0 + Math.pow(Math.E, middle[0] * -1.0));
        middle[1] = input[0] * weights[1] + input[1] * weights[11] + input[2] * weights[21] + input[3] * weights[31] + input[4] * weights[41] ;
        middle[1] = 1.0 / (1.0 + Math.pow(Math.E, middle[1] * -1.0));
        middle[2] = input[0] * weights[2] + input[1] * weights[12] + input[2] * weights[22] + input[3] * weights[32] + input[4] * weights[42] ;
        middle[2] = 1.0 / (1.0 + Math.pow(Math.E, middle[2] * -1.0));
        middle[3] = input[0] * weights[3] + input[1] * weights[13] + input[2] * weights[23] + input[3] * weights[33] + input[4] * weights[43] ;
        middle[3] = 1.0 / (1.0 + Math.pow(Math.E, middle[3] * -1.0));
        middle[4] = input[0] * weights[4] + input[1] * weights[14] + input[2] * weights[24] + input[3] * weights[34] + input[4] * weights[44] ;
        middle[4] = 1.0 / (1.0 + Math.pow(Math.E, middle[4] * -1.0));
        middle[5] = input[0] * weights[5] + input[1] * weights[15] + input[2] * weights[25] + input[3] * weights[35] + input[4] * weights[45] ;
        middle[5] = 1.0 / (1.0 + Math.pow(Math.E, middle[5] * -1.0));
        middle[6] = input[0] * weights[6] + input[1] * weights[16] + input[2] * weights[26] + input[3] * weights[36] + input[4] * weights[46] ;
        middle[6] = 1.0 / (1.0 + Math.pow(Math.E, middle[6] * -1.0));
        middle[7] = input[0] * weights[7] + input[1] * weights[17] + input[2] * weights[27] + input[3] * weights[37] + input[4] * weights[47] ;
        middle[7] = 1.0 / (1.0 + Math.pow(Math.E, middle[7] * -1.0));
        middle[8] = input[0] * weights[8] + input[1] * weights[18] + input[2] * weights[28] + input[3] * weights[38] + input[4] * weights[48] ;
        middle[8] = 1.0 / (1.0 + Math.pow(Math.E, middle[8] * -1.0));
        middle[9] = input[0] * weights[9] + input[1] * weights[19] + input[2] * weights[29] + input[3] * weights[39] + input[4] * weights[49] ;
        middle[9] = 1.0 / (1.0 + Math.pow(Math.E, middle[9] * -1.0));
        output[0] = middle[0] * weights[50] + middle[1] * weights[52] + middle[2] * weights[54] + middle[3] * weights[56] + middle[4] * weights[58] + middle[5] * weights[60] + middle[6] * weights[62] + middle[7] * weights[64] + middle[8] * weights[66] + middle[9] * weights[68] ;
        output[0] = 1.0 / (1.0 + Math.pow(Math.E, output[0] * -1.0));
        output[1] = middle[0] * weights[51] + middle[1] * weights[53] + middle[2] * weights[55] + middle[3] * weights[57] + middle[4] * weights[59] + middle[5] * weights[61] + middle[6] * weights[63] + middle[7] * weights[65] + middle[8] * weights[67] + middle[9] * weights[69] ;
        output[1] = 1.0 / (1.0 + Math.pow(Math.E, output[1] * -1.0));
        return output;
    }
}
