library('Peptides')
library('protr')

divideSeq <- function(string, prop1, prop2) {
  length <- nchar(string)
  if (length %% 3 != 0) {
    part1_length <- floor(length * prop1)
    part2_length <- floor(length * prop2)
    part1 <- substr(string, 1, part1_length)
    part2 <- substr(string, part1_length + 1, part1_length + part2_length)
    part3 <- substr(string, part1_length + part2_length + 1, length)
    return(c(part1[1], part2[1], part3[1]))
  } else {
    part_length <- length / 3
    part1 <- substr(string, 1, part_length)
    part2 <- substr(string, part_length + 1, 2 * part_length)
    part3 <- substr(string, 2 * part_length + 1, length)
    return(c(part1[1], part2[1], part3[1]))
  }
}


buildPseAAC <- function(seq,prop1,prop2,
                        lambda_max_paac,lambda_max_apaac, w = 0.05,
                        props_paac  = c("Hydrophobicity", "Hydrophilicity", "SideChainMass"),
                        props_apaac = c("Hydrophobicity", "Hydrophilicity")
) {
  PseAAC1 <- data.frame()
  PseAAC2 <- data.frame()
  
  # Pad a named vector to a fixed target-name set (fill missing with 0)
  pad_to <- function(vec, target_names) {
    out <- setNames(rep(0, length(target_names)), target_names)
    out[names(vec)] <- as.numeric(vec)
    out
  }
  
  # ------------------------------------------------------------
  # Target column names (ensure consistent dims across samples)
  # Generate once using a dummy long sequence to avoid lambda不足
  # ------------------------------------------------------------
  target_paac_names <- c(LETTERS[1:0])  # placeholder
  
  dummy_len <- max(lambda_max_paac, lambda_max_apaac) + 1
  dummy <- paste(rep("A", dummy_len), collapse = "")
  
  dummy_paac <- extractPAAC(
    dummy,
    props  = props_paac,
    lambda = lambda_max_paac,
    w      = w
  )
  target_paac_names <- sort(names(dummy_paac))
  
  dummy_apaac <- extractAPAAC(
    dummy,
    props  = props_apaac,
    lambda = lambda_max_apaac,
    w      = w
  )
  target_apaac_names <- sort(names(dummy_apaac))
  
  # ------------------------------------------------------------
  # PAAC part
  # ------------------------------------------------------------
  for (s in seq) {
    parts <- divideSeq(s, prop1, prop2)
    
    comp <- lapply(parts, function(part) {
      L   <- nchar(part)
      lam <- min(lambda_max_paac, max(1, L - 1))
      
      v <- extractPAAC(part, props = props_paac, lambda = lam, w = w)
      v <- pad_to(v, target_paac_names)
      
      as.data.frame(t(v))
    })
    
    PseAAC1 <- rbind(PseAAC1, do.call(cbind, comp))
  }
  
  # ------------------------------------------------------------
  # APAAC part
  # ------------------------------------------------------------
  for (s in seq) {
    parts <- divideSeq(s, prop1, prop2)
    
    comp2 <- lapply(parts, function(part) {
      L   <- nchar(part)
      lam <- min(lambda_max_apaac, max(1, L - 1))
      
      v <- extractAPAAC(part, props = props_apaac, lambda = lam, w = w)
      v <- pad_to(v, target_apaac_names)
      
      as.data.frame(t(v))
    })
    
    PseAAC2 <- rbind(PseAAC2, do.call(cbind, comp2))
  }
  
  # Combine
  PseAAC <- cbind(PseAAC1, PseAAC2)
  rownames(PseAAC) <- make.names(seq, unique = TRUE)
  print(paste("dimension of PseAAC is", ncol(PseAAC)))
  return(PseAAC)
}



preProcess <- function(seq){
  seq <- seq[!grepl("^>", seq)]
  seq <- seq[!grepl("[BJOUXZ]", seq)] ## remove non-standard rare residues
  seq <- Filter(function(x) nchar(x) >= 10, seq) ## filter out the ones less than 10 AAs
  seq <- Filter(function(x) nchar(x) <= 500, seq) ## filter out the ones greater than 500 AAs
  return(seq)
}

extractFeatures <- function(ampFile,nonampFile,out,split1_prop,split2_prop){
  ## INPUT: fasta files
  amp_seq <- readLines(ampFile)
  nonamp_seq <- readLines(nonampFile)
  ## Preprocess
  amp_seq <-preProcess(amp_seq)
  nonamp_seq <- preProcess(nonamp_seq)
  
  cat('Preprocessing done!\n')
  
  seq <- unlist(c(amp_seq,nonamp_seq))
  ## Get stats
  seqStats(amp_seq,AMP=T)
  seqStats(nonamp_seq,AMP=F)
  
  ## Calculate Amino Acid Composition (AAC)
  AAC <- data.frame(do.call(rbind, lapply(seq, function(i) t(data.frame(extractAAC(i))))))
  rownames(AAC) <- make.names(seq, unique=TRUE)
  AAC <- AAC[, order(names(AAC))]

  cat('AAC done!\n')

  PAAC_list <- lapply(seq, function(i) {
    PAAC1 <- extractPAAC(i, lambda = 9)
    PAAC2 <- extractAPAAC(i, lambda = 9)
    cbind(t(data.frame(PAAC1)), t(data.frame(PAAC2)))
  })
  PAAC <- do.call(rbind, PAAC_list)
  rownames(PAAC) <- seq
  cat('PAAC done!\n')
  

  PseAAC1 <- buildPseAAC(seq,split1_prop, split2_prop,
                         lambda_max_paac = 5,lambda_max_apaac = 5)
  
  cat('PseAAC done!\n')
  
  ## Assemble
  #res <- cbind(AAC,Peptide,PAAC,NAAC,PseAAC)
  res <- cbind(AAC, PAAC, PseAAC1)
  ## Labels
  labels <- c(rep(1, length(amp_seq)), rep(0, length(nonamp_seq)))
  res <- as.data.frame(scale(t(as.data.frame(res))))
  res <- as.data.frame(t(res))
  res$labels <- labels
  print(paste("the total dimension is",ncol(res)))
  cat('Finished!\n')
  
  write.csv(res,out)
  cat('Feature matrix stored in',out,'\n')
  
}




