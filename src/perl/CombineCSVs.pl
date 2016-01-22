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

my $newFile = "MapsAndWinners.csv";
open(OUTFILE, ">", $newFile) or die "File write failed on: $newFile\n";

my @keys = keys %{$hashes[0]};

foreach my $key (@keys)
{
    my $maxWins = -1;
    my $maxIndex = -1;
    my $multiple = 0;
    for(my $k = 0; $k < scalar @hashes; $k++)
    {
        if($hashes[$k]->{$key} > $maxWins)
        {
            $maxWins = $hashes[$k]->{$key};
            $maxIndex = $k;
            $multiple = 0;
        }
        elsif($hashes[$k]->{$key} == $maxWins)
        {
            $multiple = 1;
        }
    }
    if($multiple == 0)
    {
        print OUTFILE "$key,$maxIndex\n";
    }
}

close OUTFILE;
