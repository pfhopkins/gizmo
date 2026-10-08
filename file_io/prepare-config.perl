
# This file processes the configurations options in Config.sh, producing 
# two files:
#
#   GIZMO_config.h         to be included in each source file (via allvars.h)
#   compile_time_info.cc   code to be compiled in, which will print the configuration
#
# This file is based on similar code from GADGET3 by Volker Springel.
#

if(@ARGV[0])
{
   open(FILE, @ARGV[0]);
}
else
{
   open(FILE, "Config.sh");
}
open(OUTFILE, ">GIZMO_config.h");
open(COUTF,   ">compile_time_info.cc");

# include guard: eos.h and aneos.h include this file again after declarations/precompiler_logic.h has
# processed it, and without the guard that second pass would re-define every option the logic #undef'd
print OUTFILE "#ifndef GIZMO_CONFIG_H\n#define GIZMO_CONFIG_H\n";

print COUTF "#include <stdio.h>\n";
print COUTF "void output_compile_time_options(void)\n\{\n";
print COUTF "printf(\n";

while($line=<FILE>)
{
    chop $line;

    @fields = split ' ' , $line;

    if(substr($fields[0], 0, 1) ne "#")
    {
	if(length($fields[0]) > 0)
	{
	    @subfields = split '=', $fields[0];

	    print OUTFILE "#define $subfields[0] $subfields[1]\n";
            print COUTF   "\"        $fields[0]\\n\"\n";
	}
    }
}

print OUTFILE "#endif\n";
print COUTF "\"\\n\");\n";
print COUTF "\}\n";
