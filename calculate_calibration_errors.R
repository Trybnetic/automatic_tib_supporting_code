#' This script calculates the scale and offset parameters from the gt3x files
#' using GGIR's g.calibrate function. The results are stored in 
#' calibration_error.txt which is a ";" seperated file.
library(GGIR)
library(optparse)

option_list = list(
    make_option(c("-d", "--dir"), type="character", default=NULL, 
              help="folder with gt3x files", metavar="character"),
    make_option(c("-o", "--out"), type="character", default="calibration_error.txt", 
              help="output file name [default= %default]", metavar="character")
); 
 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

base_path = opt$dir
output_path = opt$out

files = list.files(path=base_path, recursive=T, pattern=".gt3x")
#files = sapply(files, function(file_path) {paste(base_path, file_path, sep="/")})
files = sapply(files, function(file_path) {paste(base_path, file_path, sep="")})

if(file.exists(output_path)){
    
    processed_files = read.csv(output_path, sep=";")$file_path
    files = setdiff(files, processed_files)
    
    print(paste("Found",length(processed_files), "processed files; start processing the remaining", length(files), "files."))

} else {
    
    write("file_path;scale1;scale2;scale3;offset1;offset2;offset3;calibrationerror_start;calibrationerror_end;message", file=output_path)
    
    print(paste("Start processing", length(files), "files."))
    
}


for (file_path in files) {
    tryCatch({
        res = g.calibrate(file_path)

        scale = paste(res$scale, collapse=";")
        offset = paste(res$offset, collapse=";")

        line = paste(file_path, scale, offset, res$cal.error.start, res$cal.error.end, res$QCmessage, sep=";", collapse=";")
        write(line, file=output_path, append=TRUE)
    }, warning = function(war) {
        print(paste("Warning:  ", war))
    }, error = function(err) {
        print(paste("Error:  ", err))     
    })
}
