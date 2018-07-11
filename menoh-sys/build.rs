extern crate pkg_config;

fn main() {
    match pkg_config::Config::new()
              .atleast_version("1.0")
              .probe("menoh") {
        Err(err) => {
            println!("cargo:warning=pkg-config failed: {}", err);
            println!("cargo:rustc-link-lib=dylib=menoh");
        }
        _ => (),
    }
}
