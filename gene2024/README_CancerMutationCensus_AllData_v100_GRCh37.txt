Description of columns in the cmc_export.tsv file

GENE_NAME - The gene name for which the data has been curated in COSMIC. In most cases this is the accepted HGNC symbol
ACCESSION_NUMBER - The transcript identifier of the gene
ONC_TSG - Role of gene in cancer
CGC_TIER - Cancer gene census Tier (for more details see https://cancer.sanger.ac.uk/census)
MUTATION_URL - URL of mutation page on the main COSMIC site
LEGACY_MUTATION_ID - Legacy mutation identifier (COSM) that represents existing COSM mutation identifiers
Mutation CDS - The change that has occurred in the nucleotide sequence. Formatting is identical to the method used for the peptide sequence
Mutation AA - The change that has occurred in the peptide sequence. Formatting is based on the recommendations made by the Human Genome Variation Society (HGVS)
AA_MUT_START - Start position of the peptide change
AA_MUT_STOP - Stop position of the peptide change (for frameshift variants, this will be the same as AA_MUT_START)
SHARED_AA - Number of mutations seen in this amino acid position
GENOMIC_WT_ALLELE_SEQ - Wild-type/reference allele in the genomic change (on the forward strand)
GENOMIC_MUT_ALLELE_SEQ - Mutant allele in the genomic change (on the forward strand)
AA_WT_ALLELE_SEQ - Wild-type/reference amino acid in the peptide change
AA_MUT_ALLELE_SEQ - Mutant amino acid in the peptide change
Mutation Description CDS - Type of mutation at the nucloetide level
Mutation Description AA - Type of mutation at the amino acid level
ONTOLOGY_MUTATION_CODE - Sequence Ontology (SO) code for the mutation
GENOMIC_MUTATION_ID - Genomic mutation identifier (COSV) to indicate the definitive position of the variant on the genome. This identifier is trackable and stable between different versions of the release
Mutation genome position GRCh37 - The genomic coordinates of the mutation on the GRCh37 assembly
Mutation genome position GRCh38 - The genomic coordinates of the mutation on the GRCh38 assembly
COSMIC_SAMPLE_TESTED - Number of samples in COSMIC tested for this mutation
COSMIC_SAMPLE_MUTATED - Number of samples in COSMIC with this mutation
DISEASE - Diseases with > 1% samples mutated (or frequency > 0.01), where disease = Primary site(tissue) / Primary histology / Sub-histology = Samples mutated / Samples tested = Frequency
WGS_DISEASE - Same as DISEASE, but for whole-genome screen data only
EXAC_AF - Allele frequency in all ExAC samples
EXAC_AFR_AF - Adjusted Alt allele frequency in African & African American ExAC samples
EXAC_AMR_AF - Adjusted Alt allele frequency in American ExAC samples
EXAC_EAS_AF - Adjusted Alt allele frequency in East Asian ExAC samples
EXAC_FIN_AF - Adjusted Alt allele frequency in Finnish ExAC samples
EXAC_NFE_AF - Adjusted Alt allele frequency in Non-Finnish European ExAC samples
EXAC_SAS_AF - Adjusted Alt allele frequency in South Asian ExAC samples
GNOMAD_EXOMES_AF - Alternative allele frequency in all gnomAD exome samples (123,136 samples)
GNOMAD_EXOMES_AFR_AF - Alternative allele frequency in the African/African American gnomAD exome samples (7,652 samples)
GNOMAD_EXOMES_AMR_AF - Alternative allele frequency in the Latino gnomAD exome samples (16,791 samples)
GNOMAD_EXOMES_ASJ_AF - Alternative allele frequency in the Ashkenazi Jewish gnomAD exome samples (4,925 samples)
GNOMAD_EXOMES_EAS_AF - Alternative allele frequency in the East Asian gnomAD exome samples (8,624 samples)
GNOMAD_EXOMES_FIN_AF - Alternative allele frequency in the Finnish gnomAD exome samples (11,150 samples)
GNOMAD_EXOMES_NFE_AF - Alternative allele frequency in the Non-Finnish European gnomAD exome samples (55,860 samples)
GNOMAD_EXOMES_SAS_AF - Alternative allele frequency in the South Asian gnomAD exome samples (15,391 samples)
GNOMAD_GENOMES_AF - Alternative allele frequency in the whole gnomAD genome samples (15,496 samples)
GNOMAD_GENOMES_AFR_AF - Alternative allele frequency in the African/African American gnomAD genome samples (4,368 samples)
GNOMAD_GENOMES_AMI_AF - Alternative allele frequency in the Amish gnomAD genome samples (450 samples)
GNOMAD_GENOMES_AMR_AF - Alternative allele frequency in the Latino gnomAD genome samples (419 samples)
GNOMAD_GENOMES_ASJ_AF - Alternative allele frequency in the Ashkenazi Jewish gnomAD genome samples (151 samples)
GNOMAD_GENOMES_EAS_AF - Alternative allele frequency in the East Asian gnomAD genome samples (811 samples)
GNOMAD_GENOMES_FIN_AF - Alternative allele frequency in the Finnish gnomAD genome samples (1,747 samples)
GNOMAD_GENOMES_NFE_AF - Alternative allele frequency in the Non-Finnish European gnomAD genome samples (7,509 samples)
GNOMAD_GENOMES_SAS_AF - Alternative allele frequency in the South Asian gnomAD genome samples (1,526 samples)
CLINVAR_CLNSIG - clinical significance as to the clinvar data set. 0 - unknown, 1 - untested, 2 - Benign, 3 - Likely benign, 4 - Likely pathogenic, 5 - Pathogenic, 6 - drug response, 7 - histocompatibility. A negative score means the the score is for the ref allele
CLINVAR_TRAIT - the trait/disease the CLINVAR_CLNSIG referring to
GERP++_RS - GERP++ RS score, the larger the score, the more conserved the site. Scores range from -12.3 to 6.17
MIN_SIFT_SCORE - Minimum SIFT score (SIFTori). Scores range from 0 to 1. The smaller the score the more likely the SNP has damaging effect
MIN_SIFT_PRED - Prediction corresponding to the minimum sift score. If SIFTori is smaller than 0.05 the corresponding nsSNV is predicted as "D(amaging)"; otherwise it is predicted as "T(olerated)"
DNDS_DISEASE_QVAL - dn/ds diseases with significant q-values (q-value < 0.05), analysed from TCGA whole-exome data in COSMIC. Diseases are classified into AML, HNSCC, NSCLC, bladder, breast, cervical, colon, endometrioid, gastric, glioma, kidney, liver, melanoma, ovary, pancreatic, prostate, sarcoma, thyroid
MUTATION_SIGNIFICANCE_TIER - Mutation significance. 1 - high significance, 2 - medium significance, 3 - low significance, Other - No predicted significance (other mutations)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
cmc_export.tsv 파일의 열에 대한 설명

GENE_NAME - 데이터가 코스믹에서 큐레이션된 유전자 이름입니다. 대부분의 경우 이 기호는 허용되는 HGNC 기호입니다
ACCESSION_NUMBER - 유전자의 전사체 식별자
ONC_TSG - 암에서 유전자의 역할
CGC_TIER - 암 유전자 인구조사 계층(자세한 내용은 https://cancer.sanger.ac.uk/census) 참조)
돌연변이_URL - 기본 코스믹 사이트의 돌연변이 페이지 URL
Legacy_MUT_ID - 기존 COSM 돌연변이 식별자를 나타내는 기존 돌연변이 식별자(COSM)
돌연변이 CDS - 뉴클레오티드 서열에서 발생한 변화입니다. 형식 지정은 펩타이드 서열에 사용되는 방법과 동일합니다
돌연변이 AA - 펩타이드 서열에서 발생한 변화. 형식 지정은 인간 게놈 변이 학회(HGVS)의 권고 사항을 기반으로 합니다
AA_MUT_START - 펩타이드 변경의 시작 위치
AA_MUT_STOP - 펩타이드 변경의 정지 위치(프레임 시프트 변형의 경우 AA_MUT_START와 동일합니다)
SHARED_AA - 이 아미노산 위치에 나타난 돌연변이 수
게놈_WT_ALLE_SEQ - 게놈 변화의 야생형/참조 대립유전자(전방향 가닥)
게놈_MUT_ALLE_SEQ - 게놈 변화의 돌연변이 대립 유전자(전방향 가닥)
AA_WT_ALLE_SEQ - 펩타이드 변화의 야생형/참조 아미노산
AA_MUT_ALLE_SEQ - 펩타이드 변화의 돌연변이 아미노산
돌연변이 설명 CDS - 뉴클로에타이드 수준의 돌연변이 유형
돌연변이 설명 AA - 아미노산 수준의 돌연변이 유형
온톨로지_뮤테이션_코드 - 돌연변이에 대한 시퀀스 온톨로지(SO) 코드
게놈_돌연변이_ID - 게놈에서 변종의 최종 위치를 나타내는 게놈 돌연변이 식별자(COSV).
돌연변이 게놈 위치 GRCh37 - GRCh37 어셈블리에서 돌연변이의 게놈 좌표
돌연변이 게놈 위치 GRCh38 - GRCh38 어셈블리에서 돌연변이의 게놈 좌표
COMISS_SAMPLE_TESTED - 이 돌연변이에 대해 테스트된 COMISS의 샘플 수
코스믹_샘플_돌연변이 - 이 돌연변이가 있는 코스믹의 샘플 수
질병 - 샘플 변이가 1% 이상인 질병(또는 빈도 > 0.01), 여기서 질병 = 원발성 부위(tissue)/원발성 조직학/하위 조직학 = 샘플 변이/검사 샘플 = 빈도
WGS_DISE - 질병과 동일하지만 전체 유전체 화면 데이터의 경우에만 해당됩니다
EXAC_AF - 모든 ExAC 샘플의 대립 유전자 빈도
EXAC_AFR_AF - 아프리카계 및 아프리카계 미국인 ExAC 샘플에서 조정된 Alt 대립유전자 빈도
EXAC_AMR_AF - 미국 ExAC 샘플에서 조정된 Alt 대립유전자 빈도
EXAC_EAS_AF - 동아시아 ExAC 샘플에서 조정된 Alt 대립유전자 빈도
EXAC_FIN_AF - 핀란드 ExAC 샘플에서 조정된 Alt 대립유전자 빈도
EXAC_NFE_AF - 비핀란드 유럽 ExAC 샘플에서 조정된 Alt 대립유전자 빈도
EXAC_SAS_AF - 남아시아 ExAC 샘플에서 조정된 Alt 대립유전자 빈도
GNOMAD_EXOMS_AF - 모든 gnomAD 엑소좀 샘플에서 대체 대립유전자 빈도(123,136개 샘플)
GNOMAD_EXOMS_AFR_AF - 아프리카/아프리카계 미국인 gnomAD 엑솜 샘플의 대체 대립유전자 빈도(7,652개 샘플)
GNOMAD_EXOMS_AMR_AF - 라틴계 gnomAD 엑소메 샘플의 대체 대립유전자 빈도(16,791개 샘플)
GNOMAD_EXOMS_ASJ_AF - 아쉬케나지 유대인 gnomAD 엑솜 샘플의 대체 대립유전자 빈도(4,925개 샘플)
GNOMAD_EXOMS_EAS_AF - 동아시아 gnomAD 엑솜 샘플의 대체 대립유전자 빈도(8,624개 샘플)
GNOMAD_EXOMES_FIN_AF - 핀란드 gnomAD 엑소메 샘플의 대체 대립유전자 빈도(11,150개 샘플)
GNOMAD_EXOMS_NFE_AF - 핀란드가 아닌 유럽의 gnomAD 엑솜 샘플에서 대체 대립유전자 빈도(55,860개 샘플)
GNOMAD_EXOMS_SAS_AF - 남아시아 gnomAD 엑솜 샘플의 대체 대립유전자 빈도(15,391개 샘플)
GNOMAD_GENOMS_AF - 전체 gnomAD 게놈 샘플에서 대체 대립유전자 빈도(15,496개 샘플)
GNOMAD_GENOMES_AFR_AF - 아프리카계/아프리카계 미국인 gnomAD 게놈 샘플의 대체 대립유전자 빈도(4,368개 샘플)
GNOMAD_GENOMES_AMI_AF - 아미쉬 gnomAD 게놈 샘플의 대체 대립유전자 빈도(450개 샘플)
GNOMAD_GENOMES_AMR_AF - 라틴계 gnomAD 게놈 샘플의 대체 대립유전자 빈도(419개 샘플)
GNOMAD_GENOMES_ASJ_AF - 아쉬케나지 유대인 gnomAD 게놈 샘플의 대체 대립유전자 빈도(151개 샘플)
GNOMAD_GENOMES_EAS_AF - 동아시아 gnomAD 게놈 샘플의 대체 대립유전자 빈도(811개 샘플)
GNOMAD_GENOMES_FIN_AF - 핀란드 gnomAD 게놈 샘플의 대체 대립유전자 빈도(1,747개 샘플)
GNOMAD_GENOMES_NFE_AF - 핀란드가 아닌 유럽 gnomAD 게놈 샘플의 대체 대립유전자 빈도(7,509개 샘플)
GNOMAD_GENOMES_SAS_AF - 남아시아 gnomAD 게놈 샘플의 대체 대립유전자 빈도(1,526개 샘플)
CLINVAR_CLNSIG - 임상적으로 유의미한 임상 데이터 세트. 0 - 알 수 없음, 1 - 테스트되지 않음, 2 - 양성, 3 - 양성 가능성, 4 - 병원성 가능성, 5 - 병원성, 6 - 약물 반응, 7 - 조직적합성. 마이너스 점수는 심판 대립유전자에 대한 점수임을 의미합니다
CLINVAR_TRAIT - CLINVAR_CLNSIG가 언급하는 형질/질병
GERP++_RS - GERP++ RS 점수가 클수록 사이트가 더 보존됩니다. 점수 범위는 -12.3 ~ 6.17입니다
MIN_SIFT_SCORE - 최소 SIFT 점수(SIFTori). 점수 범위는 0에서 1까지입니다. 점수가 작을수록 SNP는 손상 효과가 있을 가능성이 높습니다
MIN_SIFT_PRED - 최소 체수 점수에 해당하는 예측. SIFTori가 0.05보다 작으면 해당 nsSNV가 "D(amaging)"로 예측되고, 그렇지 않으면 "T(olerated)"로 예측됩니다
DNDS_DISEase_QVAL - COSMIC의 TCGA 전체 엑솜 데이터에서 분석한 유의미한 q값(q값 < 0.05)을 가진 dn/d 질병. 질병은 AML, HNSCC, NSCLC, 방광, 유방, 자궁경부, 결장, 자궁내막, 위, 신경교종, 신장, 간, 흑색종, 난소, 췌장, 전립선, 육종, 갑상선으로 분류됩니다
돌연변이_중요도_티어 - 돌연변이 유의성. 1 - 높은 유의성, 2 - 중간 유의성, 3 - 낮은 유의성, 기타 - 예측된 유의성 없음(기타 돌연변이)