library(Seurat)
library(hdf5r)
library(SeuratDisk)
rm(list = ls())

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Please provide the 'data_path' as a command-line argument.")
}

data_path <- args[1]
cat("Using data_path:", data_path, "\n")

# read 10X
dir_content <- list.dirs(path = data_path, full.names = TRUE)
for (folder in dir_content) {
  GSM <- basename(folder)
  counts <- Read10X(folder)
  obj <- CreateSeuratObject(counts,project = GSM,min.cells = 3, min.features = 200)
  obj$GEO <- basename(data_path)
  obj[["RNA"]] <- as(object = obj[["RNA"]], Class = "Assay")
  
  print(head(obj@meta.data))
  saveRDS(obj,
          file = paste0(folder,".rds")
  )
}

# read rds
rds_files <- list.files(pattern = "\\.rds$", full.names = F)
rds_files
csv_files <- list.files(pattern = "\\.csv$", full.names = F)
csv_files
# tcr filter function
TCR_filtering<-function(TCR_data){
  TCR_data<-TCR_data[TCR_data$chain == "TRB",]
  TCR_data<-TCR_data[,match(c("barcode","cdr3"),colnames(TCR_data))]
  
  ###Filter the cells that detect two or more beta chains
  dupBarcode<-(duplicated(TCR_data$barcode)|duplicated(TCR_data$barcode,fromLast = T))
  TCR_data<-TCR_data[!dupBarcode,]
  colnames(TCR_data)<-c("Barcode","Beta")
  TCR_data<-as.data.frame(TCR_data)
  TCR_data$Barcode<-as.character(TCR_data$Barcode)
  TCR_data$Beta<-as.character(TCR_data$Beta)
  
  ###Filter the cells based on the amino acid composition and length of the TCR sequence
  betaInd <- sapply(TCR_data$Beta, function(x) grepl("^[ACDEFGHIKLMNPQRSTVWY]+$", x))
  TCR_data<-TCR_data[betaInd,]
  
  bbbLength<-sapply(as.character(TCR_data$Beta),function(x){
    length(unlist(strsplit(x,split = "")))
  })
  lengthInd <- bbbLength >= 8 & bbbLength <= 25
  TCR_data<-TCR_data[lengthInd,]
  
  return(TCR_data)
}

seurat_list <- list()
for (rds_file in rds_files) {
  keyword <- sub(".rds", "", rds_file)
  csv_file <- grep(keyword, csv_files, value = TRUE)
  print(csv_file)
  
  if (length(csv_file) > 0) {
    obj <- readRDS(rds_file)
    obj$rownames <- rownames(obj@meta.data)
    tcr <- read.csv(csv_file)
    tcr_filter <- TCR_filtering(tcr)
    
    interBarcode_1<-intersect(colnames(obj),tcr_filter$Barcode)
    obj <- obj[,interBarcode_1]
    tcr_filter <- tcr_filter[match(interBarcode_1,tcr_filter$Barcode),]
    obj <- AddMetaData(obj,tcr_filter$Beta,col.name = "beta")
    
    # add GSM and GEO
    obj$orig.ident <- keyword
    obj$sample <- keyword
    
    # rename barcode
    obj@meta.data$barcode <- paste0(rownames(obj@meta.data),"--",
                                    obj@meta.data$orig.ident)
    obj <- RenameCells(obj,new.names = obj@meta.data$barcode)
    
    obj <- NormalizeData(obj)
    obj <- ScaleData(obj)
    
    seurat_list <- c(seurat_list, list(obj))
    print(head(obj@meta.data))
    saveRDS(obj,
            paste0(sub(".rds", "", rds_file), "_beta.rds")
    )
  }
}

### trans seurat to scanpy ##########
library(stringr)
library(Seurat)
library(SeuratDisk)
sub_dir_name <- basename(data_path)
obj <- readRDS(paste0(sub_dir_name, ".rds"))

options(Seurat.object.assay.version = "v3")
obj[["RNA3"]] <- as(object = obj[["RNA"]], Class = "Assay")
scRNA <- CreateSeuratObject(counts = GetAssayData(obj, assay ="RNA", layer = "counts"),
                           meta.data = obj@meta.data)

SeuratDisk::as.h5Seurat(scRNA, filename = paste0(sub_dir_name, ".h5Seurat"), 
                        save.var.features = TRUE, 
                        save.features = TRUE)
temp <- SeuratDisk::LoadH5Seurat(paste0(sub_dir_name, ".h5Seurat"))
SeuratDisk::Convert(paste0(sub_dir_name, ".h5Seurat"), dest = "h5ad")




