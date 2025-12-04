use ahash::AHashMap;
use ahash::AHashSet;
use bio::io::fasta;
use bio::io::fasta::Record;
use clap::Parser;
use rand::seq::IndexedRandom;
use rayon::prelude::*;
use std::fmt;
use std::fs::File;
use std::io;
use std::io::BufWriter;
use std::io::Write;
use std::path::PathBuf;

const NGRAM_SIZE: u8 = 8;
const ITERATION_COUNT: u8 = 100;
const SAMPLE_SIZE: u8 = 32;

fn main() -> io::Result<()> {
    let cli = Cli::parse();

    let queries = read_fasta_records(cli.queries)?;
    let targets = read_fasta_records(cli.targets)?;

    let file = File::create(cli.output)?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "query\titeration\tscore\tmax_score\tkingdom\tphylum\tclass\torder\tfamily\tgenus\tspecies\ttarget"
    )?;

    rayon::ThreadPoolBuilder::new()
        .num_threads(cli.threads)
        .build_global()
        .unwrap();

    eprintln!("LOG -- building target ngram index");
    let target_db = TargetDatabase::new(targets);

    eprintln!("LOG -- calculating scores");
    let top_hits: Vec<TopHit> = queries
        .par_iter()
        .enumerate()
        .flat_map(|(query_index, query)| {
            let mut rng = rand::rng();
            let mut top_hits: Vec<TopHit> = Vec::new();

            let mut scores = vec![0u8; target_db.records.len()];

            for iteration in 0..ITERATION_COUNT {
                // Reset scores to zero
                scores.fill(0);

                let mut highest_score = 0;

                let sampled_query_ngrams: Vec<&[u8]> =
                    subsample_ngrams(query.seq(), SAMPLE_SIZE as usize, &mut rng).collect();

                // For each sampled ngram, look up which targets contain it
                for ngram in sampled_query_ngrams {
                    if let Some(target_indices) = target_db.get_targets_for_ngram(ngram) {
                        for &target_index in target_indices {
                            scores[target_index] += 1;
                            if scores[target_index] > highest_score {
                                highest_score = scores[target_index];
                            }
                        }
                    }
                }

                let &top_hit_index = scores
                    .iter()
                    .enumerate()
                    // Get hits with the highest score
                    .filter(|(_index, score)| **score == highest_score)
                    // Keep only the index
                    .map(|(index, _score)| index)
                    .collect::<Vec<usize>>()
                    // Randomly select one
                    .choose(&mut rng)
                    .expect("should have at least one hit with the highest score");

                top_hits.push(TopHit {
                    query_index,
                    target_index: top_hit_index,
                    iteration_index: iteration,
                    score: highest_score,
                });
            }

            top_hits
        })
        .collect();

    eprintln!("LOG -- writing results");
    for top_hit in top_hits.iter() {
        let query = &queries[top_hit.query_index];
        let target = &target_db.records[top_hit.target_index];
        let target_header = record_header(target);
        let target_taxonomy = parse_taxonomy(&target_header);

        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}",
            record_header(query),
            top_hit.iteration_index,
            top_hit.score,
            SAMPLE_SIZE,
            target_taxonomy,
            target_header
        )?;
    }

    eprintln!("LOG -- done!");

    Ok(())
}

#[derive(Parser)]
#[command(version, about)]
/// Run the magical sintax algorithm on your queries and targets.
///
/// Gives more output than regular sintax!
///
struct Cli {
    /// Input queries file (e.g., amplicon sequences)
    ///
    queries: PathBuf,

    /// Input targets file (e.g., sintax DB sequences)
    ///
    targets: PathBuf,

    /// Output file name (e.g., sintaxi_output.tsv)
    ///
    output: PathBuf,

    /// Number of threads to use
    ///
    #[arg(short, long, default_value_t = 1)]
    threads: usize,
}

struct TargetDatabase {
    records: Vec<Record>,
    ngram_index: AHashMap<[u8; 8], Vec<usize>>,
}

impl TargetDatabase {
    fn new(targets: Vec<Record>) -> Self {
        let mut ngram_index: AHashMap<[u8; 8], Vec<usize>> = AHashMap::new();

        for (target_index, target) in targets.iter().enumerate() {
            let ngrams = unique_ngrams(target.seq());

            for ngram in ngrams {
                ngram_index.entry(ngram).or_default().push(target_index);
            }
        }

        Self {
            records: targets,
            ngram_index,
        }
    }

    fn get_targets_for_ngram(&self, ngram: &[u8]) -> Option<&Vec<usize>> {
        self.ngram_index.get(ngram)
    }
}

struct TopHit {
    query_index: usize,
    target_index: usize,
    iteration_index: u8,
    score: u8,
}
fn subsample_ngrams<'a>(
    seq: &'a [u8],
    sample_size: usize,
    rng: &mut impl rand::Rng,
) -> impl Iterator<Item = &'a [u8]> {
    // How many ngrams in the sequence
    let total_ngrams = seq.len().saturating_sub(7);

    // Sample indices with replacement
    let indices: Vec<usize> = (0..sample_size)
        .map(|_| rng.random_range(0..total_ngrams))
        .collect();

    indices.into_iter().map(move |i| &seq[i..i + 8])
}

fn unique_ngrams(sequence: &[u8]) -> AHashSet<[u8; 8]> {
    sequence
        .windows(NGRAM_SIZE as usize)
        .filter_map(|window| window.try_into().ok())
        .collect()
}

#[derive(Debug, Default)]
struct Taxonomy {
    kingdom: Option<String>,
    phylum: Option<String>,
    class: Option<String>,
    order: Option<String>,
    family: Option<String>,
    genus: Option<String>,
    species: Option<String>,
}

impl fmt::Display for Taxonomy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            self.kingdom.as_deref().unwrap_or(""),
            self.phylum.as_deref().unwrap_or(""),
            self.class.as_deref().unwrap_or(""),
            self.order.as_deref().unwrap_or(""),
            self.family.as_deref().unwrap_or(""),
            self.genus.as_deref().unwrap_or(""),
            self.species.as_deref().unwrap_or(""),
        )
    }
}

fn parse_taxonomy(input: &str) -> Taxonomy {
    let mut taxonomy = Taxonomy::default();

    // Find the tax= part
    if let Some(tax_part) = input.split("tax=").nth(1) {
        // Remove trailing semicolon and split by comma
        let tax_string = tax_part.trim_end_matches(';');

        for item in tax_string.split(',') {
            if let Some((prefix, value)) = item.split_once(':') {
                let value = value.to_string();
                match prefix {
                    "k" => taxonomy.kingdom = Some(value),
                    "p" => taxonomy.phylum = Some(value),
                    "c" => taxonomy.class = Some(value),
                    "o" => taxonomy.order = Some(value),
                    "f" => taxonomy.family = Some(value),
                    "g" => taxonomy.genus = Some(value),
                    "s" => taxonomy.species = Some(value),
                    _ => {}
                }
            }
        }
    }

    taxonomy
}

fn read_fasta_records(filename: PathBuf) -> io::Result<Vec<Record>> {
    let file = File::open(filename)?;
    let query_reader = fasta::Reader::new(file);
    let queries: Vec<Record> = query_reader.records().map(|x| x.unwrap()).collect();

    Ok(queries)
}

fn record_header(record: &Record) -> String {
    format!("{} {}", record.id(), record.desc().unwrap_or(""))
}
