package CreateNet;

use strict;
use warnings;

my $k;
my $t;
my $a;

my $input = $ARGV[0];
my $middle = $ARGV[1];
my $output = $ARGV[2];

my $newfile = "FeedForwardNeuralNetwork.java";

open(OUTFILE, ">", $newfile) or die "File write failed on: $newfile\n";

print OUTFILE "/**\n";
print OUTFILE " * inputs: ${input}\n";
print OUTFILE " * middle: ${middle}\n";
print OUTFILE " * output: ${output}\n";
print OUTFILE " */\n";
print OUTFILE "public class FeedForwardNeuralNetwork\n";
print OUTFILE "{\n";
print OUTFILE "    private double[] weights;\n\n";
print OUTFILE "    public FeedForwardNeuralNetwork(double[] weights)\n";
print OUTFILE "    {\n";
print OUTFILE "        this.weights = weights;\n";
print OUTFILE "    }\n\n";
print OUTFILE "    public void setWeights(double[] weights)\n";
print OUTFILE "    {\n";
print OUTFILE "        this.weights = weights;\n";
print OUTFILE "    }\n\n";
print OUTFILE "    public double[] compute(double[] input)\n";
print OUTFILE "    {\n";
print OUTFILE "        double[] weights = this.weights;\n";
print OUTFILE "        double[] middle = {";
for($k = 0; $k < $middle; $k++)
{
    print OUTFILE "0";
    if($k < $middle - 1)
    {
        print OUTFILE ",";
    }
}
print OUTFILE "};\n";
print OUTFILE "        double[] output = {";
for($k = 0; $k < $output; $k++)
{
    print OUTFILE "0";
    if($k < $output - 1)
    {
        print OUTFILE ",";
    }
}
print OUTFILE "};\n\n";
for($k = 0; $k < $middle; $k++)
{
    print OUTFILE "        middle[${k}] = ";
    for($t = 0; $t < $input; $t++)
    {
        my $index = 0;
        for($a = 0; $a < $t; $a++)
        {
            $index += $middle;
        }

        $index += $k;

        print OUTFILE "input[${t}] * weights[${index}] ";
        if($t < $input - 1)
        {
            print OUTFILE "+ ";
        }
    }
    print OUTFILE ";\n";
    print OUTFILE "        middle[${k}] = 1.0 / (1.0 + Math.pow(Math.E, middle[${k}] * -1.0));\n";
}
for($k = 0; $k < $output; $k++)
{
    print OUTFILE "        output[${k}] = ";
    for($t = 0; $t < $middle; $t++)
    {
        my $index = $input * $middle;
        for($a = 0; $a < $t; $a++)
        {
            $index += $output;
        }

        $index += $k;

        print OUTFILE "middle[${t}] * weights[${index}] ";
        if($t < $middle - 1)
        {
            print OUTFILE "+ ";
        }
    }
    print OUTFILE ";\n";
    print OUTFILE "        output[${k}] = 1.0 / (1.0 + Math.pow(Math.E, output[${k}] * -1.0));\n";
}
print OUTFILE "        return output;\n";
print OUTFILE "    }\n";
print OUTFILE "}\n";

close OUTFILE;
