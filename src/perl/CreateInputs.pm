package CreateInputs;

use strict;
use warnings;
use Data::Dumper;
use List::Util qw[min max];
use Exporter 'import';
our @EXPORT =  qw(createInputs);

sub createInputs
{
    my $map = $_[0];

    my $firstSpawn = 3000;
    my $averageSpawn = 0;
    my $averageDist = 0;
    my $size = 0;
    my $blocking = 0;

    my $num = 0;
    #First spawning round and average rounds between spawning
    foreach my $round (@{$map->{"game-map"}->{"zombieSpawnSchedule"}->{"round"}})
    {
        my $roundNum = $round->{"number"};
        if($firstSpawn > $roundNum)
        {
            $firstSpawn = $roundNum;
        }
        if($averageSpawn < $roundNum)
        {
            $averageSpawn = $roundNum;
        }
        $num++;
    }

    $averageSpawn = ($averageSpawn - $firstSpawn) / $num;



    my @aX = ();
    my @aY = ();
    my @bX = ();
    my @bY = ();

    foreach my $bot (@{$map->{"game-map"}->{"initialRobots"}->{"initial-robot"}})
    {
        if($bot->{"type"} eq "ARCHON")
        {
            if($bot->{"team"} eq "A")
            {
                push(@aX, $bot->{"originOffsetX"});
                push(@aY, $bot->{"originOffsetY"});
            }
            else
            {
                push(@bX, $bot->{"originOffsetX"});
                push(@bY, $bot->{"originOffsetY"});
            }
        }
    }

    my $archons = @aX;

    #Average distance to enemies
    for(my $k = 0; $k < $archons; $k++)
    {
        my $tempDist = 0;
        for(my $a = 0; $a < $archons; $a++)
        {
            $tempDist += (($aX[$k] - $bX[$a])**2 + ($aY[$k] - $bY[$a])**2)
        }
        $averageDist += $tempDist;
    }

    $averageDist = $averageDist / $archons;

    #Estimated map size
    my $minX = 1000;
    my $minY = 1000;
    my $maxX = 0;
    my $maxY = 0;
    for(my $k = 0; $k < $archons; $k++)
    {
        if($aX[$k] < $minX)
        {
            $minX = $aX[$k];
        }
        if($aX[$k] > $maxX)
        {
            $maxX = $aX[$k];
        }
        if($aY[$k] < $minY)
        {
            $minY = $aY[$k];
        }
        if($aY[$k] > $maxY)
        {
            $maxY = $aY[$k];
        }
        if($bX[$k] < $minX)
        {
            $minX = $bX[$k];
        }
        if($bX[$k] > $maxX)
        {
            $maxX = $bX[$k];
        }
        if($bY[$k] < $minY)
        {
            $minY = $bY[$k];
        }
        if($bY[$k] > $maxY)
        {
            $maxY = $bY[$k];
        }
    }

    $size = ($maxX - $minX) * ($maxY - $minY);

    #Enemy between buddies
    for(my $k = 0; $k < $archons; $k++)
    {
        for(my $a = $k + 1; $a < $archons; $a++)
        {
            $minX = min($aX[$k], $aX[$a]);
            $maxX = max($aX[$k], $aX[$a]);
            $minY = min($aY[$k], $aY[$a]);
            $maxY = max($aY[$k], $aY[$a]);
            for(my $i = 0; $i < $archons; $i++)
            {
                if($bX[$i] >= $minX && $bX[$i] <= $maxX && $bY[$i] >= $minY && $bY[$i] <= $maxX)
                {
                    $blocking = 1;
                }
            }
        }
    }

    #Normalize
    $firstSpawn = $firstSpawn / 500;
    $averageSpawn = $averageSpawn / 500;
    $averageDist = $averageDist / 6400;
    $size = $size / 3200;

    return ($firstSpawn, $averageSpawn, $averageDist, $size, $blocking);
}
