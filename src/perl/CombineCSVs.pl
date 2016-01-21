#!/bin/perl

use strict;
use warnings;

my @hashes;

foreach my $file (@ARGV)
{
    my $hash = {};
    open(my $fh, '<', $file) or die "Can't open $file: $!";
    while ( my $line = <$fh> )
    {
        chomp $line;
        (my $key, my $value) = split /,/, $line;
        $key =~ s/ //;
        $value =~ s/ //;
        $hash->{$key} = $value;
    }
    close $fh;

    push(@hashes, $hash);
}

my $newfile = "MapsAndWinners.csv";
open(OUTFILE, ">", $newfile) or die "File write failed on: $newfile\n";

my @keys = keys %{$hashes[0]};

foreach my $key (@keys)
{
    my $maxWins = -1;
    my $maxIndex = -1;
    for(my $k = 0; $k < scalar @hashes; $k++)
    {
        if($hashes[$k]->{$key} > $maxWins)
        {
            $maxWins = $hashes[$k]->{$key};
            $maxIndex = $k;
        }
    }
    print OUTFILE "$key,$maxIndex\n";
}

close OUTFILE;
