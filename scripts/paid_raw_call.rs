//! LOCAL paid-proof calls. Long inference keeps the same ingress request alive
//! while polling; a timeout never resubmits or charges a second request.
use candid::Principal;
use ic_agent::{Agent, export::reqwest::Url, identity::Secp256k1Identity};
use std::{fs, path::{Path, PathBuf}, process::Command, time::Duration};

const USAGE: &str = "usage: paid_raw_call [--url LOCAL_URL] [--identity-pem PEM_PATH] LOCAL_CANISTER METHOD ARGS_BIN";

struct Options {
    url: Option<String>,
    identity_pem: Option<PathBuf>,
    canister: String,
    method: String,
    argument: PathBuf,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut url = None;
        let mut identity_pem = None;
        let mut positional = Vec::new();
        let mut args = args.iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--url" => url = Some(args.next().ok_or(USAGE)?.clone()),
                "--identity-pem" => identity_pem = Some(PathBuf::from(args.next().ok_or(USAGE)?)),
                _ if arg.starts_with('-') => return Err(USAGE.into()),
                _ => positional.push(arg.clone()),
            }
        }
        if positional.len() != 3 { return Err(USAGE.into()); }
        Ok(Self { url, identity_pem, canister: positional[0].clone(),
            method: positional[1].clone(), argument: PathBuf::from(&positional[2]) })
    }
}

fn local_url(value: &str) -> Result<Url, String> {
    let url = Url::parse(value).map_err(|e| format!("invalid local URL: {e}"))?;
    let host = url.host_str().unwrap_or("").trim_start_matches('[').trim_end_matches(']');
    let loopback = host == "localhost" || host.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback());
    if !matches!(url.scheme(), "http" | "https") || !loopback {
        return Err("paid_raw_call requires a loopback local IC URL".into());
    }
    Ok(url)
}

fn network_url(root: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let output = Command::new("icp").args(["network", "status", "--json"]).current_dir(root).output()?;
    if !output.status.success() {
        return Err("Cannot read local network status; start it or specify --url".into());
    }
    let status: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    Ok(status["api_url"].as_str().ok_or("local network status has no api_url")?.into())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args == ["--help"] { println!("{USAGE}"); return Ok(()); }
    let options = Options::parse(&args)?;
    let query = match options.method.as_str() {
        "getInferenceQuote" | "getPaidInferenceConfig" | "getInferenceReceipt" | "getPaidInferenceDebug" | "balance" => true,
        "runPaidInference" | "configurePaidInference" | "retryInferenceRefund" | "runPaidInferenceWorker" | "setPaidInferenceFault" | "setPaidInferenceReference" | "forward" => false,
        _ => return Err("unsupported paid-proof method".into()),
    };
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let url = match options.url.or_else(|| std::env::var("IMAJEV_LOCAL_URL").ok()) {
        Some(url) => url,
        None => network_url(root)?,
    };
    let url = local_url(&url)?;
    let identity = options.identity_pem
        .or_else(|| std::env::var_os("IMAJEV_LOCAL_IDENTITY_PEM").map(PathBuf::from))
        .unwrap_or_else(|| root.join("artifacts/imajev-local.pem"));
    let agent = Agent::builder().with_url(url)
        .with_identity(Secp256k1Identity::from_pem_file(identity)?)
        .with_verify_query_signatures(true)
        .with_max_polling_time(Duration::from_secs(30 * 60)).build()?;
    agent.fetch_root_key().await?;
    let canister = Principal::from_text(&options.canister)?;
    let argument = fs::read(&options.argument)?;
    let reply = if query {
        agent.query(&canister,&options.method).with_arg(argument).call().await?
    } else {
        agent.update(&canister,&options.method).with_arg(argument).call_and_wait().await?
    };
    let hex: String = reply.iter().map(|v|format!("{v:02x}")).collect();
    println!("{hex}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_urls_accept_other_ports_and_ipv6_but_reject_remote_hosts() {
        for url in ["http://127.0.0.1:8100/", "http://localhost:8000/", "http://[::1]:9000/"] {
            assert!(local_url(url).is_ok(), "{url}");
        }
        for url in ["https://icp-api.io", "http://localhost.example/", "file:///tmp/ic", "invalid"] {
            assert!(local_url(url).is_err(), "{url}");
        }
    }
    #[test]
    fn explicit_options_preserve_paths_with_spaces_and_legacy_arguments() {
        let args = ["--url", "http://localhost:8100", "--identity-pem", "keys/local identity.pem",
            "aaaaa-aa", "getInferenceQuote", "args/request file.bin"].map(String::from);
        let options = Options::parse(&args).unwrap();
        assert_eq!(options.url.as_deref(), Some("http://localhost:8100"));
        assert_eq!(options.identity_pem, Some(PathBuf::from("keys/local identity.pem")));
        assert_eq!(options.argument, PathBuf::from("args/request file.bin"));
        let options = Options::parse(&args[4..]).unwrap();
        assert!(options.url.is_none() && options.identity_pem.is_none());
    }
    #[test]
    fn incomplete_or_unknown_options_are_rejected() {
        for args in [vec!["--url"], vec!["--identity-pem"], vec!["--unknown"], vec!["aaaaa-aa", "balance"]] {
            assert!(Options::parse(&args.into_iter().map(String::from).collect::<Vec<_>>()).is_err());
        }
    }
}
