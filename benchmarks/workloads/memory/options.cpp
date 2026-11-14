#include "options.h"

#include "harness/host_memory.h"

namespace computeedge::bench::storage_modes {

using namespace computeedge::bench::harness;

metal::probe::BufferKind parse_kind(const std::string& s, const char* src) {
  namespace probe = computeedge::metal::probe;
  return parse_enum(
      s,
      {probe::BufferKind::Shared, probe::BufferKind::Private, probe::BufferKind::NoCopy, probe::BufferKind::SharedCopy},
      "buffer kind", src);
}

Options parse_options(int argc, char** argv) {
  Options o;
  const char* src = kSource;
  static_cast<CommonOptions&>(o) = parse(argc, argv, src, kAllCommonFlags, [&](const std::string& f, const auto& next) {
    if (f == "--sizes")
      o.sizes = split_bytes(next());
    else if (f == "--kinds")
      o.kinds = split(next());
    else if (f == "--experiments")
      o.experiments = split(next());
    else
      return false;
    return true;
  });
  for (const auto& k : o.kinds)
    if (k != "host") parse_kind(k, src);
  for (size_t s : o.sizes)
    if (s == 0 || s % page_size() != 0) usage(src, "sizes must be non-zero multiples of the page size (16K)");
  return o;
}

}  // namespace computeedge::bench::storage_modes
