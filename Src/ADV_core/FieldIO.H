#ifndef FIELDIO_H_
#define FIELDIO_H_

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Gpu.H>

#include <fstream>
#include <string>
#include <sstream>

// ---------------------------------------------------------------------------
// writeNpy: gather component 0 of mf onto the IO rank and write it as a NumPy
// .npy file, shape (nx[, ny[, nz]]), Fortran order (x fastest), so that
// np.load(f)[i, j, k] is the value at AMReX index (i, j, k).
//
// Covers the full index range of mf's IndexType (a face-centred field in a
// periodic direction has one extra, duplicated, face). Assumes a
// little-endian host.
// ---------------------------------------------------------------------------

inline void writeNpy (const amrex::MultiFab& mf, const amrex::Geometry& geom,
                      const std::string& fname)
{
    const amrex::Box dom = amrex::convert(geom.Domain(), mf.ixType());

    const int io = amrex::ParallelDescriptor::IOProcessorNumber();
    amrex::BoxArray ba1(dom);
    amrex::DistributionMapping dm1(amrex::Vector<int>{io});
    amrex::MultiFab one(ba1, dm1, 1, 0, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    one.ParallelCopy(mf, 0, 0, 1);
    amrex::Gpu::streamSynchronize();

    if (!amrex::ParallelDescriptor::IOProcessor()) { return; }

    // ---- header -----------------------------------------------------------
    const amrex::IntVect len = dom.length();
    std::ostringstream shape;
    shape << "(";
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { shape << len[d] << ", "; }
    shape << ")";

    const char* descr = (sizeof(amrex::Real) == 8) ? "<f8" : "<f4";
    std::string dict = std::string("{'descr': '") + descr
                     + "', 'fortran_order': True, 'shape': " + shape.str() + ", }";
    // magic(6) + version(2) + len(2) + dict + padding + '\n' : multiple of 64
    std::size_t total = 10 + dict.size() + 1;
    dict.append((64 - total % 64) % 64, ' ');
    dict.push_back('\n');
    const auto hlen = static_cast<unsigned short>(dict.size());

    std::ofstream ofs(fname, std::ios::binary);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ofs.good(), "writeNpy: cannot open output file");
    ofs.write("\x93NUMPY", 6);
    const char ver[2] = {1, 0};
    ofs.write(ver, 2);
    const char hl[2] = {static_cast<char>(hlen & 0xff), static_cast<char>((hlen >> 8) & 0xff)};
    ofs.write(hl, 2);
    ofs.write(dict.data(), static_cast<std::streamsize>(dict.size()));

    // ---- data, x fastest ------------------------------------------------------
    for (amrex::MFIter mfi(one); mfi.isValid(); ++mfi)
    {
        auto const& a = one.const_array(mfi);
        const auto lo = amrex::lbound(dom);
        const auto hi = amrex::ubound(dom);
        for (int k = lo.z; k <= hi.z; ++k) {
        for (int j = lo.y; j <= hi.y; ++j) {
        for (int i = lo.x; i <= hi.x; ++i) {
            const amrex::Real v = a(i,j,k);
            ofs.write(reinterpret_cast<const char*>(&v), sizeof(amrex::Real));
        }}}
    }
}

#endif
