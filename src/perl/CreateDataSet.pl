use strict;
use warnings;
use XML::Simple;
use CreateInputs;
use Data::Dumper;

my $winnersFile = $ARGV[0];
my $mapsDir = $ARGV[1];

my $newfile = "dataset.csv";
open(OUTFILE, ">", $newfile) or die "File write failed on: $newfile\n";

open(my $fh, '<', $winnersFile) or die "Can't open $winnersFile: $!";

while ( my $line = <$fh> )
{
    chomp $line;
    (my $xml, my $winner) = split /,/, $line;
    my $file = "$mapsDir$xml";
    my $map = XMLin($file);
    my @inputs = createInputs($map);
    foreach my $input (@inputs)
    {
        print OUTFILE "$input,";
    }
    print OUTFILE "$winner\n";
}

