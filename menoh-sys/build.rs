fn main() {
    match pkg_config::Config::new()
        .atleast_version("1.1")
        .probe("menoh")
    {
        Ok(_) => (),
        Err(err) => {
            println!("cargo:warning=pkg-config failed: {}", err);
            println!("cargo:rustc-link-lib=dylib=menoh");
        }
    }
}
