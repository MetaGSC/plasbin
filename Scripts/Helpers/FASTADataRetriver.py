from Bio import SeqIO


def createDNAbertFileFromFASTA(size, inputFile, outputFile, k, batch_size):
    total = 0
    for record in SeqIO.parse(inputFile, 'fasta'):
        if(total == size):
            break
        with open(outputFile, 'a') as f:
            m = str(record.seq)
            kmer = [m[x:x+k] for x in range(len(m)+1-k)]
            batch_count = int(len(kmer)/batch_size)+1
            for i in range(batch_count):
                batch_kmers = kmer[(batch_size*i):(batch_size*(i+1))]
                kmerline = "  ".join(batch_kmers)
                f.write(kmerline+"\n")
            total += 1
            if(total % 100 == 0):
                print(total)


def createMiniFASTAFile(startIndex, size, inputFile, outputFile):
    total = 0
    newFileSeqs = []
    for record in SeqIO.parse(inputFile, 'fasta'):
        total += 1
        if(total < startIndex):
            continue
        if(total == startIndex+size):
            break
        if(total < startIndex):
            break
        if(total % 100 == 0):
            print(total)
    SeqIO.write(newFileSeqs, outputFile, 'fasta')


def createMiniFastaBatches(batchSize, inputFile, outputDir):
    total = 0
    batchCount = 0
    newFileSeqs = []
    for record in SeqIO.parse(inputFile, 'fasta'):
        newFileSeqs.append(record)
        total += 1
        if(total == batchSize):
            outputFile = outputDir + '/DNAML_bacterial_batch' + \
                str(batchCount) + '.fasta'
            print(f'Writing batch {batchCount} of size {batchSize}')
            SeqIO.write(newFileSeqs, outputFile, 'fasta')
            total = 0
            batchCount += 1
            newFileSeqs = []
        if (total % 100 == 0):
            print(f'{total} sequences of batch {batchCount} read')
    outputFile = outputDir + '/DNAML_bacterial_batch'+str(batchCount)+'.fasta'
    SeqIO.write(newFileSeqs, outputFile, 'fasta')


def createMiniFASTAFileWithSpecificSequence(ID, inputFile, outputFile):
    total = 0
    newFileSeqs = []
    for record in SeqIO.parse(inputFile, 'fasta'):
        total += 1
        if(record.id == ID):
            newFileSeqs.append(record)
            break
        if(total % 100 == 0):
            print(total)
    SeqIO.write(newFileSeqs, outputFile, 'fasta')


def createMixedFastaFile(plasmidStartIndex, chromosomeStartIndex, size, plasmidRatio, plasmidInputFile, chromosomeInputFile, outputFile):
    total = 0
    newFileSeqs = []
    print('plasmid')
    for record in SeqIO.parse(plasmidInputFile, 'fasta'):
        newFileSeqs.append(record)
        total += 1
        if(total == plasmidStartIndex+int(size*plasmidRatio)):
            break
        if(total < plasmidStartIndex):
            break
        if(total % 100 == 0):
            print(total)
    print('chromosome')
    total = 0
    for record in SeqIO.parse(chromosomeInputFile, 'fasta'):
        newFileSeqs.append(record)
        total += 1
        if(total == chromosomeStartIndex+int(size*(1-plasmidRatio))):
            break
        if(total < chromosomeStartIndex):
            break
        if(total % 100 == 0):
            print(total)
    SeqIO.write(newFileSeqs, outputFile, 'fasta')


def countSequences(inputFile):
    total = 0
    length = 0
    newFileSeqs = []
    for record in SeqIO.parse(inputFile, 'fasta'):
        total += 1
        cat = len(record.seq)
        if(cat > length):
            length = cat
        if(total % 100 == 0):
            print(total)
    print("All sequences", total)
    print("Max length", length)
