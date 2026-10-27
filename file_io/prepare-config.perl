
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
# processed it, and without the guard that second pass would re-define every option the logic #undef'd.
# A standalone build (e.g. test/aneos_unit) defines GIZMO_config_H to compile without a Config.
print OUTFILE "#ifndef GIZMO_config_H\n#define GIZMO_config_H\n";

print COUTF "#include <stdio.h>\n";
print COUTF "void output_compile_time_options(void)\n\{\n";
print COUTF "printf(\n";

%option_names = ();
$literal_options = "";
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
	    $option_names{$subfields[0]} = 1;
	    ($escaped = $fields[0]) =~ s/(["\\])/\\$1/g;
	    $literal_options .= "\"$escaped\\n\"\n";
	}
    }
}

print OUTFILE "#endif\n";
print COUTF "\"\\n\");\n";
print COUTF "\}\n";

# The effective settings: every name in this Config, every object-like macro defined by precompiler_logic.h
# (where the *_DEFAULTS cascades and derived options live), and every option named in Template_Config.sh,
# each reported with its value if it is defined once all of those are applied.
foreach $source ("declarations/precompiler_logic.h", "Template_Config.sh")
{
    if(open(SRC, $source))
    {
        while($line=<SRC>)
        {
            if($source =~ /\.h$/) {if($line =~ /^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)(\s|$)/) {$option_names{$1} = 1;}}
            else {if($line =~ /^\s*#*\s*([A-Z][A-Z0-9_]{2,})(\s|=|$)/) {$option_names{$1} = 1;}}
        }
        close(SRC);
    }
}

print COUTF "\n#include <string.h>\n#include <stdlib.h>\n";
print COUTF "#include \"GIZMO_config.h\"\n#include \"declarations/precompiler_logic.h\"\n";
print COUTF "#define GIZMO_OPTION_VALUE_STRING(...) #__VA_ARGS__\n";
print COUTF "#define GIZMO_OPTION_VALUE(...) GIZMO_OPTION_VALUE_STRING(__VA_ARGS__)\n";
print COUTF "static const char *effective_option_table[][2] = {\n";
foreach $name (sort keys %option_names)
{
    print COUTF "#ifdef $name\n    {\"$name\", GIZMO_OPTION_VALUE($name)},\n#endif\n";
}
print COUTF "    {NULL, NULL}\n};\n";
print COUTF "const char *compile_time_config_literal(void) {return\n$literal_options\"\";}\n";
print COUTF <<'END_OF_FUNCTIONS';
/* the effective options, one per line, NAME or NAME=value, sorted by name */
const char *compile_time_effective_options(void)
{
    static char *text = NULL;
    if(text) {return text;}
    size_t length = 1;
    for(int k = 0; effective_option_table[k][0]; k++) {length += strlen(effective_option_table[k][0]) + strlen(effective_option_table[k][1]) + 2;}
    text = (char *) malloc(length); text[0] = 0;
    for(int k = 0; effective_option_table[k][0]; k++)
    {
        strcat(text, effective_option_table[k][0]);
        if(effective_option_table[k][1][0]) {strcat(text, "="); strcat(text, effective_option_table[k][1]);}
        strcat(text, "\n");
    }
    return text;
}
/* 64-bit FNV-1a hash of the effective options: equal for builds with the same effective settings */
unsigned long long compile_time_effective_options_fnv1a64(void)
{
    unsigned long long hash = 14695981039346656037ULL;
    for(const unsigned char *c = (const unsigned char *) compile_time_effective_options(); *c; c++) {hash ^= *c; hash *= 1099511628211ULL;}
    return hash;
}
const char *compile_time_compiler_version(void)
{
#ifdef __VERSION__
    return __VERSION__;
#else
    return "unknown";
#endif
}
END_OF_FUNCTIONS
