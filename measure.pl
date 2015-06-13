#!/usr/bin/env perl
#
# Copyright (C) 2015 Kurt Kanzenbach <kurt@kmk-computers.de>
#
# This script gathers the performance results from gather program and
# collects them in text files, which can be used for gnuplot later on.
#

use warnings;
use strict;
use FindBin;
use Data::Dumper;

my (%supported_archs, %supported_cxx, %perf_data, @archs);

sub func_arch_to_real_arch
{
    my ($func_arch) = @_;

    return "sse2"    if $func_arch eq "sse";
    return "sse4.1"  if $func_arch eq "sse4";
    return "avx"     if $func_arch eq "avx";
    return "avx2"    if $func_arch eq "avx2";
    return "avx512f" if $func_arch eq "avx512";

    die("Sth. went wrong...");
}

sub get_perf_data
{
    my ($output, $cxx, $arch) = @_;

    foreach my $line (split /\n/, $output) {
        next unless $line =~ /^Perf: /m;
        my ($func, $cycles) = $line =~ /Perf: (\w+): (\d+)/m;
        die("Program output has unexpected format: $line") unless ($func || $cycles);
        my ($_arch) = $func =~ /^(?:gather|scatter)_([a-zA-Z0-9]+)/;
        die("Program output has unexpected format: $line") unless ($_arch);
        next unless func_arch_to_real_arch($_arch) eq $arch;
        push(@{$perf_data{$cxx}{$arch}{$func}}, $cycles);
    }

    return;
}


sub benchmark
{
    my ($iter) = @_;

    # clear
    `rm -rf data`;
    die("Rm failed") if $?;

    # get supported compiler
    get_supported_compilers();

    # get supported archtitectures for compiling
    get_supported_architectures();

    foreach my $cxx (sort keys %supported_cxx) {
        foreach my $arch (sort keys %supported_archs) {
            # build with specified arch
            `make clean 2>/dev/null`;
            die("Make clean failed") if $?;
            `make ADTFLAGS='-m$arch -DWITH_PERF' CXX=$cxx 2>/dev/null`;
            die("Make \"make ADTFLAGS='-m$arch -DWITH_PERF' CXX=$cxx\"failed") if $?;

            # execute and save data
            for my $i (0..$iter-1) {
                my ($output);

                $output = `taskset 1 ./gather < input`;
                die("Gather program failed") if $?;
                get_perf_data($output, $cxx, $arch);
            }
        }
    }

    # build gnuplot files
    build_data_files();

    # build statistics and print it
    print_statistics();
}

sub get_supported_architectures
{
    my ($fh, $line);

    open($fh, "<", "/proc/cpuinfo");

    while ($line = <$fh>) {
        my ($flags);
        next unless ($flags) = $line =~ /flags\s*: (.*)/;
        chomp $flags;
        $supported_archs{"sse2"}    = 1 if $flags =~ /\s+sse2\s+/;
        $supported_archs{"sse4.1"}  = 1 if $flags =~ /\s+sse4_1\s+/;
        $supported_archs{"avx"}     = 1 if $flags =~ /\s+avx\s+/;
        $supported_archs{"avx2"}    = 1 if $flags =~ /\s+avx2\s+/;
        $supported_archs{"avx512f"} = 1 if $flags =~ /\s+avx512f\s+/;
        last;
    }

    close $fh;

    return;
}

sub get_supported_compilers
{
    `which g++ >/dev/null 2>&1`;
    $supported_cxx{"g++"}     = 1 unless $?;
    `which clang++ >/dev/null 2>&1`;
    $supported_cxx{"clang++"} = 1 unless $?;
    `which icpc >/dev/null 2>&1`;
    $supported_cxx{"icpc"}    = 1 unless $?;

    return;
}

sub avg
{
    my ($arr_ref) = @_;
    my ($res);

    $res = 0;
    $res += $_ for @$arr_ref;
    $res /= scalar @$arr_ref;

    return $res;
}

sub cl_avg
{
    my ($arr_ref) = @_;
    my ($median, $q25, $q75, $num, $q_diff);
    my (@new, $res);

    $num = @$arr_ref;

    $median = $num / 2 - 1;
    $q25    = $arr_ref->[$median / 2];
    $q75    = $arr_ref->[$median / 2 + $num / 2];
    $median = $arr_ref->[$median];
    $q_diff = $q75 - $q25;

    # strip
    foreach my $val (@$arr_ref) {
        push(@new, $val) unless ($val < $q25 - $q_diff ||
                                 $val > $q75 + $q_diff);
    }

    $res = 0;
    $res += $_ for (@new);
    $res /= scalar @new;

    return $res;
}

sub build_data_files
{
    `rm -rf data`;
    die("Rm failed") if $?;

    unless (-d "data") {
        mkdir "data" || die("Cannot create directory data: $!");
    }

    foreach my $cxx (sort keys %perf_data) {
        foreach my $arch (sort keys %{$perf_data{$cxx}}) {
            my ($fh_gather, $fh_scatter, @funcs, @funcs_gather, @funcs_scatter, $i);

            open($fh_gather, ">", "data/data_gather_$arch" . "_$cxx")
                || die("Cannot open data gather file: $!");
            open($fh_scatter, ">", "data/data_scatter_$arch" . "_$cxx")
                || die("Cannot open data scatter file: $!");

            @funcs         = sort keys %{$perf_data{$cxx}{$arch}};
            @funcs_gather  = grep { $_ =~ /gather/ } @funcs;
            @funcs_scatter = grep { $_ =~ /scatter/ } @funcs;
            print $fh_gather  "# run @funcs_gather\n";
            print $fh_scatter "# run @funcs_scatter\n";
            for $i (0..@{$perf_data{$cxx}{$arch}{$funcs[0]}}-1) {
                my (@line_gather, @line_scatter);
                foreach my $func (@funcs_gather) {
                    push(@line_gather, $perf_data{$cxx}{$arch}{$func}->[$i]);
                }
                foreach my $func (@funcs_scatter) {
                    push(@line_scatter, $perf_data{$cxx}{$arch}{$func}->[$i]);
                }
                print $fh_gather  "$i @line_gather\n";
                print $fh_scatter "$i @line_scatter\n";
            }

            close $fh_gather;
            close $fh_scatter;
        }
    }
}

sub print_statistics
{
    print "Statistics:\n";
    foreach my $cxx (sort keys %perf_data) {
        print "  Compiler: $cxx -- " . `$cxx --version | head -n 1`;
        foreach my $arch (sort keys %{$perf_data{$cxx}}) {
            print "    Architecture: $arch\n";
            foreach my $func (sort keys %{$perf_data{$cxx}{$arch}}) {
                my ($avg, $cl_avg, $min, $max);
                @{$perf_data{$cxx}{$arch}{$func}} = sort { $a <=> $b } @{$perf_data{$cxx}{$arch}{$func}};
                $min    = @{$perf_data{$cxx}{$arch}{$func}}[0];
                $max    = @{$perf_data{$cxx}{$arch}{$func}}[(scalar @{$perf_data{$cxx}{$arch}{$func}}) - 1];
                $avg    = avg($perf_data{$cxx}{$arch}{$func});
                $cl_avg = cl_avg($perf_data{$cxx}{$arch}{$func});
                print "      $func: $min $max $avg $cl_avg\n";
            }
        }
    }

    return;
}

chdir("$FindBin::RealBin") || die("Cannot change directory to $FindBin::RealBin: $!");

die("usage: $0 <iterations>") unless ($ARGV[0]);

benchmark($ARGV[0]);

exit 0;
