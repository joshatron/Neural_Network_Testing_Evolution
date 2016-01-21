use strict;
use warnings;
use XML::Simple;
use CreateInputs;

opendir(DIR, $ARGV[0]) || die "Can't open directory!";
my @list = readdir(DIR);
closedir(DIR);

foreach my $xml (@list)
{
    if($xml =~ m/arena\.xml$/)
    {
        my $file = "$ARGV[0]$xml";
        my $map = XMLin($file);
        createInputs($map);
    }
}

